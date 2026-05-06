#include "gw/cli.hpp"
#include "gw/gw.hpp"
#include "gw/io.hpp"
#include "gw/linalg.hpp"
#include "gw/memory_footprint.hpp"
#include "gw/npy.hpp"
#include "gw/profiling.hpp"
#include "gw/types.hpp"
#ifdef GW_HAS_BLAS_LAPACK_BACKEND
  #include "gw/linalg_blas_lapack.hpp"
#endif

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
		// call Cli, read parameters
        const gw::Cli cli = gw::parse_cli(argc, argv);
        const std::string log_path = gw::join_path(cli.output_dir, "gw.out");
        std::ofstream log_file(log_path);
        if (!log_file) {
            throw std::runtime_error("Could not open log file: " + log_path);
        }
        const gw::CoutTee cout_tee(log_file);

        std::cout << "\n--- Begin of C++20 G0W0 Calculation ---\n";
        std::cout << "Loading data from: " << cli.input_dir << '\n';

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

        std::cout << "  - Vxc MO basis shape: (" << input.vxc_mo.rows() << ", " << input.vxc_mo.cols() << ")\n";
        std::cout << "  - Fermi energy (eV): " << input.fermi_energy * gw::kHartreeToEv << '\n';
        std::cout << "  - Number of occupied orbitals: " << input.nocc << '\n';
        std::cout << "  - Number of virtual orbitals: " << nvirt << '\n';
        std::cout << "  - Number of MOs: " << nmo << '\n';
        std::cout << "  - Number of particle-hole states: " << n_ph << '\n';
        std::cout << "  - ERI MO shape: (" << input.eri_mo.dim0() << ", " << input.eri_mo.dim1() << ", "
                  << input.eri_mo.dim2() << ", " << input.eri_mo.dim3() << ")\n";

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
        } else {
            throw std::runtime_error("Unknown --linalg-backend value: " + cli.linalg_backend + ". Supported values: reference, blas-lapack.");
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

		// estimate memory usage
        if (cli.print_memory_footprint) {
            const gw::MemoryFootprint footprint = gw::estimate_memory_footprint(input, settings.num_freq_points_total);
            gw::print_memory_footprint_report(footprint);
        }

		// run g0w0
        const gw::GwResult result = gw::run_g0w0(input, settings);

		// Pade fitting
        const std::size_t output_state = settings.selected_state_0based.value_or(0);
        gw::output_self_energy_before_pade(cli.output_dir+"/E_c_before_Pade.out", result.sigma_c_im_points, output_state);
        std::vector<gw::Complex> state_sigma(result.omegas.size());
        for (std::size_t f = 0; f < result.omegas.size(); ++f) {
            state_sigma[f] = result.sigma_c_im_points(output_state, f);
        }
        gw::GwTimings timings = result.timings;
        const auto output_pade_start = gw::ProfilingClock::now();
        gw::output_self_energy_after_pade(cli.output_dir+"/E_c.out", state_sigma, result.omegas, settings.num_pade_params, 19836, input.fermi_energy);
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
