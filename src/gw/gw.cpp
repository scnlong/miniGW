#include "gw/gw.hpp"

#include "gw/frequency_grids.hpp"
#include "gw/qp_solver.hpp"
#include "gw/sigma.hpp"
#include "gw/sigma_c_driver.hpp"
#include "workspace/pq_ph_panel.hpp"
#include "workspace/screening_workspace.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace gw {
namespace {

using Clock = std::chrono::steady_clock;

#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
std::atomic_bool g_openmp_kernel_loops_enabled{true};

bool kernel_loops_enabled() noexcept {
    return g_openmp_kernel_loops_enabled.load(std::memory_order_relaxed);
}
#else
bool kernel_loops_enabled() noexcept {
    return false;
}
#endif

double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

bool root_rank(const GwSettings& settings) noexcept {
    return settings.execution.mpi_rank == 0;
}

std::vector<std::size_t> selected_states(std::size_t nmo, const std::optional<std::size_t>& selected) {
    if (selected.has_value()) {
        if (*selected >= nmo) {
            throw std::runtime_error("Selected state is out of range");
        }
        return {*selected};
    }
    std::vector<std::size_t> states(nmo);
    for (std::size_t i = 0; i < nmo; ++i) {
        states[i] = i;
    }
    return states;
}

void validate_input_shapes(const OrbitalSpace& orbitals, const MolecularIntegrals& integrals) {
    if (integrals.nmo() != orbitals.nmo()) {
        throw std::runtime_error("Input shape mismatch: integrals.nmo() != orbitals.nmo()");
    }
}

void configure_runtime_kernel_loops(bool enabled) noexcept {
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
    g_openmp_kernel_loops_enabled.store(enabled, std::memory_order_relaxed);
    set_sigma_openmp_kernel_loops(enabled);
    workspace::set_host_screening_openmp_kernel_loops(enabled);
    workspace::set_pq_ph_panel_openmp_kernel_loops(enabled);
#else
    (void)enabled;
#endif
}

std::vector<Complex> make_imaginary_frequency_grid(const std::vector<double>& omegas) {
    std::vector<Complex> omega_im(omegas.size());
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for schedule(static) if(kernel_loops_enabled())
#endif
    for (std::size_t i = 0; i < omegas.size(); ++i) {
        omega_im[i] = Complex{0.0, omegas[i]};
    }
    return omega_im;
}

void print_execution_summary(const GwSettings& settings, const linalg::Backend& linalg_backend) {
    if (!root_rank(settings)) {
        return;
    }

    const auto caps = linalg_backend.capabilities();
    std::cout << "Linear algebra backend: " << linalg_backend.name() << '\n';
    if (caps.uses_cosma_pxgemm) {
        std::cout << "COSMA mode: selected explicit prefixed pxgemm backend. "
                  << "Distributed GEMM calls use cosma_pzgemm_; "
                  << "pzgetrf_/pzgetrs_/BLACS remain provided by ScaLAPACK.\n";
        if (caps.external_provider_may_use_gpu) {
            std::cout << "COSMA GPU capability: the linked COSMA provider was built with GPU-capable support; "
                      << "actual GPU use is a COSMA runtime decision and should be verified with nvidia-smi/Nsight.\n";
        }
    }
    std::cout << "Frequency parallel mode: " << to_string(settings.execution.frequency_parallel_mode) << '\n';
    std::cout << "OpenMP kernel loops: " << (settings.execution.openmp_kernel_loops ? "enabled" : "disabled") << '\n';
    std::cout << "MPI rank/size: " << settings.execution.mpi_rank << " / " << settings.execution.mpi_size << '\n';
    if (settings.execution.mpi_size > 1 &&
        (settings.execution.openmp_kernel_loops || caps.uses_internal_threads)) {
        std::cout << "Warning: MPI ranks are combined with local threaded work. "
                  << "For a first MPI run, prefer OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1.\n";
    } else if (caps.uses_internal_threads &&
               std::getenv("OPENBLAS_NUM_THREADS") == nullptr &&
               std::getenv("MKL_NUM_THREADS") == nullptr) {
        std::cout << "Warning: BLAS/LAPACK backend may use internal threads. "
                  << "Set OPENBLAS_NUM_THREADS or MKL_NUM_THREADS for reproducible CPU usage.\n";
    }
}

void validate_frequency_execution(const GwSettings& settings, const linalg::Backend& linalg_backend) {
    const auto caps = linalg_backend.capabilities();
    if (settings.execution.frequency_parallel_mode != FrequencyParallelMode::OpenMP) {
        return;
    }
    if (!caps.thread_safe) {
        throw std::runtime_error("OpenMP frequency parallelism requires a thread-safe linear algebra backend.");
    }
    if (caps.distributed_mpi || caps.uses_device_memory) {
        throw std::runtime_error("OpenMP frequency parallelism is restricted to local CPU backends. Use MPI frequency distribution or a backend-managed GPU task queue for distributed/device backends.");
    }
    if (caps.uses_internal_threads && root_rank(settings)) {
        std::cout << "Warning: OpenMP frequency parallelism with a threaded BLAS/LAPACK backend can oversubscribe CPUs. Prefer OPENBLAS_NUM_THREADS=1.\n";
    }
}

} // namespace

void add_timings(GwTimings& dst, const GwTimings& src) {
    dst.total_wall_seconds += src.total_wall_seconds;
    dst.exchange_seconds += src.exchange_seconds;
    dst.build_mapping_seconds += src.build_mapping_seconds;
    dst.build_inv_v_seconds += src.build_inv_v_seconds;
    dst.build_pi0_seconds += src.build_pi0_seconds;
    dst.invert_epsilon_seconds += src.invert_epsilon_seconds;
    dst.sigma_c_seconds += src.sigma_c_seconds;
    dst.sigma_c_wall_seconds += src.sigma_c_wall_seconds;
    dst.mpi_reduce_seconds += src.mpi_reduce_seconds;
    dst.pade_seconds += src.pade_seconds;
}

GwResult run_g0w0(const GwInput& input, const GwSettings& settings) {
    const auto run_start = Clock::now();
    configure_runtime_kernel_loops(settings.execution.openmp_kernel_loops);

    OrbitalSpace orbitals(input.mo_energy, input.nocc, input.fermi_energy);
    MolecularIntegrals integrals(input.eri_mo, input.vxc_mo);
    validate_input_shapes(orbitals, integrals);

    const ParticleHoleBasis ph_basis(orbitals);
    const linalg::Backend& linalg_backend = settings.linalg_backend ? *settings.linalg_backend : linalg::reference_backend();
    validate_frequency_execution(settings, linalg_backend);
    print_execution_summary(settings, linalg_backend);

    auto [omegas, weights] = generate_transformed_legendre_grid(settings.num_freq_points_total);
    if (root_rank(settings)) {
        std::cout << "Total number of Pade parameters: " << settings.num_pade_params << '\n';
        std::cout << "Total number of transformed Gauss-Legendre frequency points: " << settings.num_freq_points_total << '\n';
    }

    GwResult result;
    result.omegas = omegas;
    result.weights = weights;

    auto start = Clock::now();
    result.sigma_x = calculate_exchange(orbitals, integrals);
    result.timings.exchange_seconds = elapsed_seconds(start, Clock::now());

    start = Clock::now();
    const std::vector<Complex> omega_im = make_imaginary_frequency_grid(omegas);
    result.timings.build_mapping_seconds = elapsed_seconds(start, Clock::now());

    result.sigma_c_im_points = MatrixComplex(orbitals.nmo(), settings.num_freq_points_total, Complex{0.0, 0.0});
    const auto states = selected_states(orbitals.nmo(), settings.selected_state_0based);

    if (root_rank(settings)) {
        std::cout << "\n--- Starting Sigma_c(iw) calculation... ---\n";
    }
    start = Clock::now();
    compute_sigma_c_with_backend(orbitals,
                                 integrals,
                                 ph_basis,
                                 omega_im,
                                 weights,
                                 states,
                                 settings,
                                 linalg_backend,
                                 result.sigma_c_im_points,
                                 result.timings);
    result.timings.sigma_c_wall_seconds = elapsed_seconds(start, Clock::now());

    if (root_rank(settings)) {
        std::cout << "Sigma_c(iw) calculation complete.\n";
    }

    start = Clock::now();
    result.qp_energy = solve_qp_energies(orbitals,
                                         integrals,
                                         result.sigma_x,
                                         result.sigma_c_im_points,
                                         omega_im,
                                         states,
                                         settings);
    result.timings.pade_seconds = elapsed_seconds(start, Clock::now());
    result.timings.total_wall_seconds = elapsed_seconds(run_start, Clock::now());

    return result;
}

} // namespace gw
