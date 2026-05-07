#include "gw/cli.hpp"
#include "gw/gw.hpp"
#include "gw/io.hpp"
#include "gw/linalg.hpp"
#include "gw/memory_footprint.hpp"
#include "gw/mpi_context.hpp"
#ifdef GW_HAS_BLAS_LAPACK_BACKEND
#include "gw/linalg_blas_lapack.hpp"
#endif
#ifdef GW_HAS_SCALAPACK_BACKEND
#include "gw/linalg_scalapack.hpp"
#endif
#ifdef GW_HAS_COSMA_BACKEND
#include "gw/linalg_cosma.hpp"
#endif
#ifdef GW_HAS_CUDA_BACKEND
#include "gw/linalg_cublas.hpp"
#endif
#include "gw/npy.hpp"
#include "gw/profiling.hpp"
#include "gw/types.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

gw::FrequencyParallelMode choose_frequency_parallel_mode(const gw::Cli& cli, const gw::MpiContext& mpi) {
    if (cli.frequency_parallel == "serial") {
        if (mpi.size() > 1) {
            throw std::runtime_error("--frequency-parallel serial must not be launched with multiple MPI ranks; run without mpirun or use --frequency-parallel mpi");
        }
        return gw::FrequencyParallelMode::Serial;
    }
    if (cli.frequency_parallel == "mpi") {
        if (mpi.size() == 1) {
            throw std::runtime_error("--frequency-parallel mpi requires running with more than one MPI rank");
        }
#ifndef GW_ENABLE_MPI
        throw std::runtime_error("--frequency-parallel mpi requires a build configured with -DGW_ENABLE_MPI=ON");
#else
        return gw::FrequencyParallelMode::MPI;
#endif
    }
    if (cli.frequency_parallel == "openmp") {
#ifndef GW_ENABLE_OPENMP_FREQUENCY_PARALLEL
        throw std::runtime_error("--frequency-parallel openmp requires -DGW_ENABLE_OPENMP_FREQUENCY_PARALLEL=ON");
#else
        return gw::FrequencyParallelMode::OpenMP;
#endif
    }

    // auto: MPI frequency distribution if launched with multiple ranks; otherwise serial.
    if (mpi.size() > 1) {
#ifdef GW_ENABLE_MPI
        return gw::FrequencyParallelMode::MPI;
#else
        throw std::runtime_error("Internal error: MPI size > 1 but GW_ENABLE_MPI is not defined");
#endif
    }
    return gw::FrequencyParallelMode::Serial;
}

std::size_t frequency_workspace_replicas(gw::FrequencyParallelMode mode) {
    if (mode == gw::FrequencyParallelMode::OpenMP) {
#ifdef GW_ENABLE_OPENMP_FREQUENCY_PARALLEL
#ifdef _OPENMP
        return static_cast<std::size_t>(omp_get_max_threads());
#else
        return 1;
#endif
#else
        return 1;
#endif
    }
    return 1;
}

bool choose_openmp_kernel_loops(const gw::Cli& cli, gw::FrequencyParallelMode frequency_mode) {
    if (cli.kernel_parallel == "serial") {
        return false;
    }
    if (cli.kernel_parallel == "openmp") {
#ifndef GW_ENABLE_OPENMP_KERNEL_LOOPS
        throw std::runtime_error("--kernel-parallel openmp requires -DGW_ENABLE_OPENMP_KERNEL_LOOPS=ON");
#else
        return true;
#endif
    }

    // In auto mode, avoid multiplying MPI ranks by OpenMP kernel threads.
    if (frequency_mode == gw::FrequencyParallelMode::MPI) {
        return false;
    }
#ifdef GW_ENABLE_OPENMP_KERNEL_LOOPS
    return true;
#else
    return false;
#endif
}

} // namespace

int main(int argc, char** argv) {
    try {
		// call Cli, read parameters
        gw::MpiContext mpi(argc, argv);
        const gw::Cli cli = gw::parse_cli(argc, argv);

        std::ofstream log_file;
        std::unique_ptr<gw::CoutTee> cout_tee;
        if (mpi.root()) {
            const std::string log_path = gw::join_path(cli.output_dir, "gw.out");
            log_file.open(log_path);
            if (!log_file) {
                throw std::runtime_error("Could not open log file: " + log_path);
            }
            cout_tee = std::make_unique<gw::CoutTee>(log_file);
        }

        if (mpi.root()) {
            std::cout << "\n--- Begin of C++20 G0W0 Calculation ---\n";
            std::cout << "Loading data from: " << cli.input_dir << '\n';
        }

		// read inputs from PySCF
        const auto read_input_start = gw::ProfilingClock::now();
        gw::GwInput input;
        input.eri_mo = gw::read_tensor4_npy_f64(gw::join_path(cli.input_dir, "eri_mo.npy"));
        input.mo_energy = gw::read_vector_npy_f64(gw::join_path(cli.input_dir, "mo_energy.npy"));
        input.vxc_mo = gw::read_matrix_npy_f64(gw::join_path(cli.input_dir, "vxc_mo.npy"));
        input.nocc = static_cast<std::size_t>(gw::read_int_text(gw::join_path(cli.input_dir, "nocc.txt")));
        input.fermi_energy = gw::read_double_text(gw::join_path(cli.input_dir, "fermi_energy.txt"));
        const double read_input_seconds = gw::elapsed_seconds(read_input_start, gw::ProfilingClock::now());

        const std::size_t nmo = input.mo_energy.size();
        if (input.nocc > nmo) {
            throw std::runtime_error("Input error: nocc exceeds number of molecular orbitals");
        }
        const std::size_t nvirt = nmo - input.nocc;
        const std::size_t n_ph = input.nocc * nvirt;

        if (mpi.root()) {
            std::cout << "  - Vxc MO basis shape: (" << input.vxc_mo.rows() << ", " << input.vxc_mo.cols() << ")\n";
            std::cout << "  - Fermi energy (eV): " << input.fermi_energy * gw::kHartreeToEv << '\n';
            std::cout << "  - Number of occupied orbitals: " << input.nocc << '\n';
            std::cout << "  - Number of virtual orbitals: " << nvirt << '\n';
            std::cout << "  - Number of MOs: " << nmo << '\n';
            std::cout << "  - Number of particle-hole states: " << n_ph << '\n';
            std::cout << "  - ERI MO shape: (" << input.eri_mo.dim0() << ", " << input.eri_mo.dim1() << ", "
                      << input.eri_mo.dim2() << ", " << input.eri_mo.dim3() << ")\n";
        }

		// select linear algebra backend
        gw::GwSettings settings;
        if (cli.linalg_backend == "reference") {
            settings.linalg_backend = gw::linalg::make_reference_backend();
        } else if (cli.linalg_backend == "blas-lapack") {
#ifdef GW_HAS_BLAS_LAPACK_BACKEND
            settings.linalg_backend = gw::linalg::make_blas_lapack_backend();
#else
            throw std::runtime_error("This executable was built without the BLAS/LAPACK backend. Reconfigure with -DGW_ENABLE_BLAS_LAPACK=ON, or use --linalg-backend reference.");
#endif
        } else if (cli.linalg_backend == "scalapack") {
#ifdef GW_HAS_SCALAPACK_BACKEND
            settings.linalg_backend = gw::linalg::make_scalapack_backend();
#else
            throw std::runtime_error("This executable was built without the ScaLAPACK backend interface. Reconfigure with -DGW_ENABLE_SCALAPACK=ON.");
#endif
        } else if (cli.linalg_backend == "cosma") {
#ifdef GW_HAS_COSMA_BACKEND
            settings.linalg_backend = gw::linalg::make_cosma_backend();
#else
            throw std::runtime_error("This executable was built without the COSMA backend interface. Reconfigure with -DGW_ENABLE_COSMA=ON.");
#endif
        } else if (cli.linalg_backend == "cublas") {
#ifdef GW_HAS_CUDA_BACKEND
            settings.linalg_backend = gw::linalg::make_cublas_backend();
#else
            throw std::runtime_error("This executable was built without the cuBLAS/cuSolver backend interface. Reconfigure with -DGW_ENABLE_CUDA=ON.");
#endif
        } else {
            throw std::runtime_error("Unknown --linalg-backend value: " + cli.linalg_backend + ". Supported values: reference, blas-lapack, scalapack, cosma, cublas.");
        }

		// setup parameters
        settings.num_freq_points_total = cli.freq_points;
        settings.num_pade_params = cli.pade_params;
        settings.eta = cli.eta;
        if (cli.selected_state_1based.has_value()) {
            if (*cli.selected_state_1based == 0) {
                throw std::runtime_error("--state must be 1-based and positive");
            }
            settings.selected_state_0based = *cli.selected_state_1based - 1;
        } else {
            settings.selected_state_0based.reset();
        }

        settings.execution.frequency_parallel_mode = choose_frequency_parallel_mode(cli, mpi);
        settings.execution.mpi_rank = static_cast<std::size_t>(mpi.rank());
        settings.execution.mpi_size = static_cast<std::size_t>(mpi.size());
        settings.execution.openmp_kernel_loops = choose_openmp_kernel_loops(cli, settings.execution.frequency_parallel_mode);
        settings.execution.openmp_frequency_parallel = settings.execution.frequency_parallel_mode == gw::FrequencyParallelMode::OpenMP;
        settings.execution.frequency_workspace_replicas = frequency_workspace_replicas(settings.execution.frequency_parallel_mode);

		// estimate memory usage
        if (cli.print_memory_footprint && mpi.root()) {
            const gw::MemoryFootprint footprint = gw::estimate_memory_footprint(input, settings);
            gw::print_memory_footprint_report(footprint);
        }

		// run g0w0
        const gw::GwResult result = gw::run_g0w0(input, settings);

        if (!mpi.root()) {
            return EXIT_SUCCESS;
        }

		// Pade fitting
        const std::size_t output_state = settings.selected_state_0based.value_or(0);
        gw::output_self_energy_before_pade(cli.output_dir + "/E_c_before_Pade.out", result.sigma_c_im_points, output_state);
        std::vector<gw::Complex> state_sigma(result.omegas.size());
        for (std::size_t f = 0; f < result.omegas.size(); ++f) {
            state_sigma[f] = result.sigma_c_im_points(output_state, f);
        }
        gw::GwTimings timings = result.timings;
        const auto output_pade_start = gw::ProfilingClock::now();
        gw::output_self_energy_after_pade(cli.output_dir + "/E_c.out", state_sigma, result.omegas, settings.num_pade_params, 19836, input.fermi_energy);
        timings.pade_seconds += gw::elapsed_seconds(output_pade_start, gw::ProfilingClock::now());

		// output profiling
        gw::output_profiling_baseline(read_input_seconds, timings);

		// print out energies
        std::cout << "\n--- G0W0 Calculation Summary ---\n";
        if (settings.selected_state_0based.has_value()) {
            gw::output_final_energy_table(input.mo_energy, result.sigma_x, input.vxc_mo, result.qp_energy, {*settings.selected_state_0based});
        } else {
            std::vector<std::size_t> all_states(nmo);
            for (std::size_t i = 0; i < nmo; ++i) {
                all_states[i] = i;
            }
            gw::output_final_energy_table(input.mo_energy, result.sigma_x, input.vxc_mo, result.qp_energy, all_states);
        }
        std::cout << "\n--- End of C++20 G0W0 Calculation ---\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
