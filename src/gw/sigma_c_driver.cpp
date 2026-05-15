#include "gw/sigma_c_driver.hpp"

#include "gw/sigma.hpp"
#include "mpi_context.hpp"
#include "workspace/pq_ph_panel.hpp"
#include "workspace/screening_workspace.hpp"
#ifdef GW_HAS_CUDA_BACKEND
#include "linalg/linalg_cublas.hpp"
#include "workspace/device_pq_ph_panel.hpp"
#include "workspace/device_screening_workspace.hpp"
#endif
#ifdef GW_HAS_SCALAPACK_BACKEND
#include "workspace/distributed_screening_workspace.hpp"
#endif
#ifdef GW_HAS_COSMA_BACKEND
#include "workspace/cosma_distributed_screening_workspace.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace gw {
namespace {

using Clock = std::chrono::steady_clock;

#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
std::atomic_bool g_driver_openmp_kernel_loops_enabled{true};

bool kernel_loops_enabled() noexcept {
    return g_driver_openmp_kernel_loops_enabled.load(std::memory_order_relaxed);
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

MatrixComplex to_complex(const MatrixReal& in) {
    MatrixComplex out(in.rows(), in.cols(), Complex{0.0, 0.0});
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for collapse(2) schedule(static) if(kernel_loops_enabled())
#endif
    for (std::size_t i = 0; i < in.rows(); ++i) {
        for (std::size_t j = 0; j < in.cols(); ++j) {
            out(i, j) = Complex{in(i, j), 0.0};
        }
    }
    return out;
}

void compute_sigma_c_frequency(std::size_t f_n,
                               const OrbitalSpace& orbitals,
                               const ParticleHoleBasis& ph_basis,
                               const workspace::HostScreeningWorkspace& screening,
                               const workspace::PqPhPanelView& pq_ph_view,
                               const std::vector<Complex>& omega_im,
                               const std::vector<double>& weights,
                               const std::vector<std::size_t>& states,
                               const GwSettings& settings,
                               MatrixComplex& sigma_c_im_points,
                               GwTimings& local_timings) {
    const Complex omega_n_im = omega_im[f_n];
    std::vector<Complex> current_sigma(orbitals.nmo(), Complex{0.0, 0.0});

    for (std::size_t f_prime = 0; f_prime < settings.num_freq_points_total; ++f_prime) {
        const Complex omega_prime_im = omega_im[f_prime];

        auto phase_start = Clock::now();
        const auto pi0_diag = calculate_pi0_ph_diag(omega_prime_im, ph_basis, settings.eta);
        local_timings.build_pi0_seconds += elapsed_seconds(phase_start, Clock::now());

        phase_start = Clock::now();
        const MatrixComplex w_c_ph = screening.compute_w_c(pi0_diag);
        local_timings.invert_epsilon_seconds += elapsed_seconds(phase_start, Clock::now());

        phase_start = Clock::now();
        const std::size_t panel_size = std::max<std::size_t>(1, settings.contraction_panel_size);
        for (const auto p_idx : states) {
            for (std::size_t k0 = 0; k0 < orbitals.nmo(); k0 += panel_size) {
                const std::size_t width = std::min(panel_size, orbitals.nmo() - k0);
                MatrixReal pk_panel = pq_ph_view.make_panel(p_idx, k0, width);

                const std::vector<Complex> w_minus_v_panel =
                    screening.quadratic_forms_panel(w_c_ph, pk_panel);

                for (std::size_t kk = 0; kk < width; ++kk) {
                    const std::size_t k_idx = k0 + kk;
                    const Complex g0_denominator = omega_n_im + orbitals.fermi_energy() - orbitals.energy(k_idx);
                    const Complex g0_term = g0_denominator /
                        (g0_denominator * g0_denominator - omega_prime_im * omega_prime_im);
                    current_sigma[p_idx] -= g0_term * w_minus_v_panel[kk] * weights[f_prime];
                }
            }
        }
        local_timings.sigma_c_seconds += elapsed_seconds(phase_start, Clock::now());
    }

    for (const auto p : states) {
        sigma_c_im_points(p, f_n) = current_sigma[p] / kPi;
    }
}

#ifdef GW_HAS_CUDA_BACKEND
void compute_sigma_c_frequency_device(std::size_t f_n,
                                      const OrbitalSpace& orbitals,
                                      const ParticleHoleBasis& ph_basis,
                                      workspace::DeviceScreeningWorkspace& screening,
                                      workspace::DevicePqPhPanelView& pq_ph_view,
                                      const std::vector<Complex>& omega_im,
                                      const std::vector<double>& weights,
                                      const std::vector<std::size_t>& states,
                                      const GwSettings& settings,
                                      MatrixComplex& sigma_c_im_points,
                                      GwTimings& local_timings) {
    const Complex omega_n_im = omega_im[f_n];
    std::vector<Complex> current_sigma(orbitals.nmo(), Complex{0.0, 0.0});

    for (std::size_t f_prime = 0; f_prime < settings.num_freq_points_total; ++f_prime) {
        const Complex omega_prime_im = omega_im[f_prime];

        auto phase_start = Clock::now();
        const auto pi0_diag = calculate_pi0_ph_diag(omega_prime_im, ph_basis, settings.eta);
        local_timings.build_pi0_seconds += elapsed_seconds(phase_start, Clock::now());

        phase_start = Clock::now();
        screening.compute_w_c(pi0_diag);
        local_timings.invert_epsilon_seconds += elapsed_seconds(phase_start, Clock::now());

        phase_start = Clock::now();
        const std::size_t panel_size = std::max<std::size_t>(1, settings.contraction_panel_size);
        for (const auto p_idx : states) {
            for (std::size_t k0 = 0; k0 < orbitals.nmo(); k0 += panel_size) {
                const std::size_t width = std::min(panel_size, orbitals.nmo() - k0);
                const workspace::DeviceComplexPanelView pk_panel =
                    pq_ph_view.fill_panel(p_idx, k0, width);

                const std::vector<Complex> w_minus_v_panel =
                    screening.quadratic_forms_panel(pk_panel);

                for (std::size_t kk = 0; kk < width; ++kk) {
                    const std::size_t k_idx = k0 + kk;
                    const Complex g0_denominator = omega_n_im + orbitals.fermi_energy() - orbitals.energy(k_idx);
                    const Complex g0_term = g0_denominator /
                        (g0_denominator * g0_denominator - omega_prime_im * omega_prime_im);
                    current_sigma[p_idx] -= g0_term * w_minus_v_panel[kk] * weights[f_prime];
                }
            }
        }
        local_timings.sigma_c_seconds += elapsed_seconds(phase_start, Clock::now());
    }

    for (const auto p : states) {
        sigma_c_im_points(p, f_n) = current_sigma[p] / kPi;
    }
}

void compute_sigma_c_device_screening(const OrbitalSpace& orbitals,
                                      const ParticleHoleBasis& ph_basis,
                                      workspace::DeviceScreeningWorkspace& screening,
                                      workspace::DevicePqPhPanelView& pq_ph_view,
                                      const std::vector<Complex>& omega_im,
                                      const std::vector<double>& weights,
                                      const std::vector<std::size_t>& states,
                                      const GwSettings& settings,
                                      MatrixComplex& sigma_c_im_points,
                                      GwTimings& timings) {
    const std::size_t nfreq = settings.num_freq_points_total;
    const std::size_t rank = settings.execution.mpi_rank;
    const std::size_t size = settings.execution.mpi_size;
    const bool mpi_frequency = settings.execution.frequency_parallel_mode == FrequencyParallelMode::MPI;
    const std::size_t local_total = mpi_frequency
        ? (rank < nfreq ? ((nfreq - 1U - rank) / size + 1U) : 0U)
        : nfreq;
    std::size_t local_completed = 0;

    auto chunk_start = Clock::now();
    for (std::size_t f_n = mpi_frequency ? rank : 0; f_n < nfreq; f_n += (mpi_frequency ? size : 1U)) {
        compute_sigma_c_frequency_device(f_n,
                                         orbitals,
                                         ph_basis,
                                         screening,
                                         pq_ph_view,
                                         omega_im,
                                         weights,
                                         states,
                                         settings,
                                         sigma_c_im_points,
                                         timings);
        ++local_completed;
        if (root_rank(settings) && !mpi_frequency &&
            ((f_n + 1) % 10 == 0 || f_n + 1 == nfreq)) {
            const auto now = Clock::now();
            std::cout << "  completed CUDA device-resident frequency " << (f_n + 1) << " / "
                      << nfreq << " in " << elapsed_seconds(chunk_start, now) << " s\n";
            chunk_start = now;
        } else if (mpi_frequency && (local_completed == local_total || local_completed % 10U == 0U)) {
            const auto now = Clock::now();
            std::cout << "  rank " << settings.execution.mpi_rank
                      << " completed " << local_completed << " / " << local_total
                      << " assigned CUDA frequencies on device " << settings.execution.cuda_device_id
                      << " (last global frequency " << (f_n + 1) << " / " << nfreq << ") in "
                      << elapsed_seconds(chunk_start, now) << " s\n";
            chunk_start = now;
        }
    }
}
#endif

#ifdef GW_HAS_SCALAPACK_BACKEND
template <class DistributedScreeningWorkspaceT>
void compute_sigma_c_frequency_distributed(std::size_t f_n,
                                           const OrbitalSpace& orbitals,
                                           const ParticleHoleBasis& ph_basis,
                                           DistributedScreeningWorkspaceT& screening,
                                           const workspace::PqPhPanelView& pq_ph_view,
                                           const std::vector<Complex>& omega_im,
                                           const std::vector<double>& weights,
                                           const std::vector<std::size_t>& states,
                                           const GwSettings& settings,
                                           MatrixComplex& sigma_c_im_points,
                                           GwTimings& local_timings) {
    const Complex omega_n_im = omega_im[f_n];
    std::vector<Complex> current_sigma(orbitals.nmo(), Complex{0.0, 0.0});

    for (std::size_t f_prime = 0; f_prime < settings.num_freq_points_total; ++f_prime) {
        const Complex omega_prime_im = omega_im[f_prime];

        auto phase_start = Clock::now();
        const auto pi0_diag = calculate_pi0_ph_diag(omega_prime_im, ph_basis, settings.eta);
        local_timings.build_pi0_seconds += elapsed_seconds(phase_start, Clock::now());

        phase_start = Clock::now();
        screening.compute_w_c(pi0_diag);
        local_timings.invert_epsilon_seconds += elapsed_seconds(phase_start, Clock::now());

        phase_start = Clock::now();
        const std::size_t panel_size = std::max<std::size_t>(1, settings.contraction_panel_size);
        for (const auto p_idx : states) {
            for (std::size_t k0 = 0; k0 < orbitals.nmo(); k0 += panel_size) {
                const std::size_t width = std::min(panel_size, orbitals.nmo() - k0);
                MatrixReal pk_panel = pq_ph_view.make_panel(p_idx, k0, width);

                const std::vector<Complex> w_minus_v_panel = screening.quadratic_forms_panel(pk_panel);

                for (std::size_t kk = 0; kk < width; ++kk) {
                    const std::size_t k_idx = k0 + kk;
                    const Complex g0_denominator = omega_n_im + orbitals.fermi_energy() - orbitals.energy(k_idx);
                    const Complex g0_term = g0_denominator /
                        (g0_denominator * g0_denominator - omega_prime_im * omega_prime_im);
                    current_sigma[p_idx] -= g0_term * w_minus_v_panel[kk] * weights[f_prime];
                }
            }
        }
        local_timings.sigma_c_seconds += elapsed_seconds(phase_start, Clock::now());
    }

    const bool grouped_frequency_parallel =
        settings.execution.frequency_parallel_mode == FrequencyParallelMode::MPI;
    const bool contributes_to_world_result =
        !grouped_frequency_parallel || settings.execution.frequency_group_rank == 0;

    if (contributes_to_world_result) {
        for (const auto p : states) {
            sigma_c_im_points(p, f_n) = current_sigma[p] / kPi;
        }
    }
}

template <class DistributedScreeningWorkspaceT>
void compute_sigma_c_distributed_screening(const OrbitalSpace& orbitals,
                                           const ParticleHoleBasis& ph_basis,
                                           DistributedScreeningWorkspaceT& screening,
                                           const workspace::PqPhPanelView& pq_ph_view,
                                           const std::vector<Complex>& omega_im,
                                           const std::vector<double>& weights,
                                           const std::vector<std::size_t>& states,
                                           const GwSettings& settings,
                                           MatrixComplex& sigma_c_im_points,
                                           GwTimings& timings) {
    const bool grouped_frequency_parallel =
        settings.execution.frequency_parallel_mode == FrequencyParallelMode::MPI;
    const std::size_t group_id = grouped_frequency_parallel ? settings.execution.frequency_group_id : 0U;
    const std::size_t num_groups = grouped_frequency_parallel ? settings.execution.num_frequency_groups : 1U;
    const bool group_root = settings.execution.frequency_group_rank == 0;

    std::size_t local_completed = 0;
    const std::size_t local_total =
        group_id < settings.num_freq_points_total
            ? ((settings.num_freq_points_total - 1U - group_id) / num_groups + 1U)
            : 0U;

    auto chunk_start = Clock::now();
    for (std::size_t f_n = group_id;
         f_n < settings.num_freq_points_total;
         f_n += num_groups) {
        compute_sigma_c_frequency_distributed(f_n,
                                              orbitals,
                                              ph_basis,
                                              screening,
                                              pq_ph_view,
                                              omega_im,
                                              weights,
                                              states,
                                              settings,
                                              sigma_c_im_points,
                                              timings);

        ++local_completed;
        const bool print_progress =
            group_root && (local_completed == local_total || local_completed % 10U == 0U);
        if (print_progress) {
            const auto now = Clock::now();
            std::cout << "  group " << group_id << " completed " << local_completed
                      << " / " << local_total << " assigned distributed frequencies"
                      << " (last global frequency " << (f_n + 1) << " / "
                      << settings.num_freq_points_total << ") in "
                      << elapsed_seconds(chunk_start, now) << " s\n";
            chunk_start = now;
        }
    }
}
#endif

void compute_sigma_c_serial_or_mpi(const OrbitalSpace& orbitals,
                                   const ParticleHoleBasis& ph_basis,
                                   const workspace::HostScreeningWorkspace& screening,
                                   const workspace::PqPhPanelView& pq_ph_view,
                                   const std::vector<Complex>& omega_im,
                                   const std::vector<double>& weights,
                                   const std::vector<std::size_t>& states,
                                   const GwSettings& settings,
                                   MatrixComplex& sigma_c_im_points,
                                   GwTimings& timings) {
    const std::size_t nfreq = settings.num_freq_points_total;
    const std::size_t rank = settings.execution.mpi_rank;
    const std::size_t size = settings.execution.mpi_size;
    const bool mpi_frequency = settings.execution.frequency_parallel_mode == FrequencyParallelMode::MPI;

    auto chunk_start = Clock::now();
    for (std::size_t f_n = mpi_frequency ? rank : 0; f_n < nfreq; f_n += (mpi_frequency ? size : 1)) {
        compute_sigma_c_frequency(f_n,
                                  orbitals,
                                  ph_basis,
                                  screening,
                                  pq_ph_view,
                                  omega_im,
                                  weights,
                                  states,
                                  settings,
                                  sigma_c_im_points,
                                  timings);

        if (root_rank(settings) && !mpi_frequency && ((f_n + 1) % 10 == 0 || f_n + 1 == nfreq)) {
            const auto now = Clock::now();
            std::cout << "  completed frequency " << (f_n + 1) << " / " << nfreq
                      << " in " << elapsed_seconds(chunk_start, now) << " s\n";
            chunk_start = now;
        }
    }
}

void compute_sigma_c_openmp_frequency(const OrbitalSpace& orbitals,
                                      const ParticleHoleBasis& ph_basis,
                                      const workspace::HostScreeningWorkspace& screening,
                                      const workspace::PqPhPanelView& pq_ph_view,
                                      const std::vector<Complex>& omega_im,
                                      const std::vector<double>& weights,
                                      const std::vector<std::size_t>& states,
                                      const GwSettings& settings,
                                      MatrixComplex& sigma_c_im_points,
                                      GwTimings& timings) {
#if defined(GW_ENABLE_OPENMP_FREQUENCY_PARALLEL)
#pragma omp parallel
    {
        GwTimings local_timings{};
#pragma omp for schedule(dynamic, 1)
        for (std::size_t f_n = 0; f_n < settings.num_freq_points_total; ++f_n) {
            compute_sigma_c_frequency(f_n,
                                      orbitals,
                                      ph_basis,
                                      screening,
                                      pq_ph_view,
                                      omega_im,
                                      weights,
                                      states,
                                      settings,
                                      sigma_c_im_points,
                                      local_timings);
        }
#pragma omp critical
        add_timings(timings, local_timings);
    }
#else
    (void)orbitals;
    (void)ph_basis;
    (void)screening;
    (void)pq_ph_view;
    (void)omega_im;
    (void)weights;
    (void)states;
    (void)settings;
    (void)sigma_c_im_points;
    (void)timings;
    throw std::runtime_error("OpenMP frequency parallelism requested, but this executable was not built with -DGW_ENABLE_OPENMP_FREQUENCY_PARALLEL=ON");
#endif
}

} // namespace

void compute_sigma_c_with_backend(const OrbitalSpace& orbitals,
                                  const MolecularIntegrals& integrals,
                                  const ParticleHoleBasis& ph_basis,
                                  const std::vector<Complex>& omega_im,
                                  const std::vector<double>& weights,
                                  const std::vector<std::size_t>& states,
                                  const GwSettings& settings,
                                  const linalg::Backend& linalg_backend,
                                  MatrixComplex& sigma_c_im_points,
                                  GwTimings& timings) {
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
    g_driver_openmp_kernel_loops_enabled.store(settings.execution.openmp_kernel_loops, std::memory_order_relaxed);
#endif

    const auto caps = linalg_backend.capabilities();
    const workspace::PqPhPanelView pq_ph_view(integrals, ph_basis);
    if (root_rank(settings)) {
        std::cout << "pq_ph panel view: nmo=" << pq_ph_view.nmo()
                  << ", n_ph=" << pq_ph_view.nph()
                  << ", panel size=" << settings.contraction_panel_size << '\n';
    }

#ifdef GW_HAS_CUDA_BACKEND
    if (std::string_view{linalg_backend.name()}.find("cublas") != std::string_view::npos) {
        if (settings.execution.frequency_parallel_mode == FrequencyParallelMode::OpenMP) {
            throw std::runtime_error("Device-resident CUDA screening supports serial or MPI frequency parallelism, not OpenMP frequency parallelism.");
        }
        GwSettings cuda_settings = settings;
        cuda_settings.execution.cuda_device_id = linalg::set_cuda_device_for_local_rank(
            settings.execution.mpi_local_rank,
            settings.execution.tasks_per_gpu);
        if (root_rank(cuda_settings)) {
            std::cout << "Using CUDA device-resident screening workspace for V_ph/epsilon/W_c.\n";
            std::cout << "ERI is still host-replicated; CUDA pq panels use an auto-selected full-resident or streaming panel source.\n";
            std::cout << "CUDA MPI rank-to-GPU policy: tasks-per-gpu=" << cuda_settings.execution.tasks_per_gpu
                      << ", local rank/size=" << cuda_settings.execution.mpi_local_rank << " / "
                      << cuda_settings.execution.mpi_local_size << ", selected device="
                      << cuda_settings.execution.cuda_device_id << "\n";
        }
        auto device_start = Clock::now();
        MatrixReal v_ph = calculate_v_ph_matrix(integrals, ph_basis);
        workspace::DeviceScreeningWorkspace screening(v_ph);
        workspace::DevicePqPhPanelView device_pq_ph_view(integrals, ph_basis, cuda_settings.contraction_panel_size);
        timings.build_inv_v_seconds = elapsed_seconds(device_start, Clock::now());
        if (root_rank(cuda_settings)) {
            std::cout << "CUDA screening workspace estimated device allocation after setup: "
                      << static_cast<double>(screening.estimated_device_bytes()) / (1024.0 * 1024.0)
                      << " MiB\n";
            std::cout << "CUDA pq-panel source mode: " << device_pq_ph_view.storage_mode_name() << '\n';
            std::cout << "CUDA pq-panel source estimated device allocation: "
                      << static_cast<double>(device_pq_ph_view.estimated_device_bytes()) / (1024.0 * 1024.0)
                      << " MiB\n";
            std::cout << "CUDA pq-panel source estimated pinned host staging: "
                      << static_cast<double>(device_pq_ph_view.estimated_host_pinned_bytes()) / (1024.0 * 1024.0)
                      << " MiB\n";
        }
        compute_sigma_c_device_screening(orbitals, ph_basis, screening, device_pq_ph_view, omega_im, weights,
                                         states, cuda_settings, sigma_c_im_points, timings);
        if (cuda_settings.execution.frequency_parallel_mode == FrequencyParallelMode::MPI &&
            cuda_settings.execution.mpi_size > 1) {
            auto reduce_start = Clock::now();
            mpi_allreduce_sum_in_place(sigma_c_im_points.data());
            timings.mpi_reduce_seconds += elapsed_seconds(reduce_start, Clock::now());
        }
        return;
    }
#endif

#ifdef GW_HAS_SCALAPACK_BACKEND
    if (caps.distributed_mpi) {
        if (root_rank(settings)) {
            std::cout << "Using distributed ScaLAPACK-style screening workspace for V_ph/epsilon/W_c.\n";
            if (caps.uses_cosma_pxgemm) {
                std::cout << "Distributed GEMM provider: COSMA prefixed PBLAS ABI, calling cosma_pzgemm_.\n";
            } else {
                std::cout << "Distributed GEMM provider: ScaLAPACK/PBLAS PZGEMM, calling pzgemm_.\n";
            }
            std::cout << "ERI is still replicated; pq_ph is generated as contraction panels.\n";
            if (settings.execution.frequency_parallel_mode == FrequencyParallelMode::MPI) {
                std::cout << "ScaLAPACK frequency groups: " << settings.execution.num_frequency_groups
                          << " groups, up to " << settings.execution.scalapack_ranks_per_group
                          << " ranks/group.\n";
            } else {
                std::cout << "ScaLAPACK communicator: all " << settings.execution.mpi_size
                          << " ranks cooperate on each frequency.\n";
            }
        }
        auto distributed_start = Clock::now();
        if (caps.uses_cosma_pxgemm) {
#ifdef GW_HAS_COSMA_BACKEND
            workspace::CosmaDistributedScreeningWorkspace screening(integrals,
                                                                    ph_basis,
                                                                    64,
                                                                    settings.execution.frequency_group_size);
            timings.build_inv_v_seconds = elapsed_seconds(distributed_start, Clock::now());
            compute_sigma_c_distributed_screening(orbitals, ph_basis, screening, pq_ph_view, omega_im, weights,
                                                  states, settings, sigma_c_im_points, timings);
#else
            throw std::runtime_error(
                "COSMA backend was selected, but miniGW was built without GW_HAS_COSMA_BACKEND.");
#endif
        } else {
            workspace::DistributedScreeningWorkspace screening(integrals,
                                                               ph_basis,
                                                               64,
                                                               settings.execution.frequency_group_size);
            timings.build_inv_v_seconds = elapsed_seconds(distributed_start, Clock::now());
            compute_sigma_c_distributed_screening(orbitals, ph_basis, screening, pq_ph_view, omega_im, weights,
                                                  states, settings, sigma_c_im_points, timings);
        }
        if (settings.execution.frequency_parallel_mode == FrequencyParallelMode::MPI && settings.execution.mpi_size > 1) {
            auto reduce_start = Clock::now();
            mpi_allreduce_sum_in_place(sigma_c_im_points.data());
            timings.mpi_reduce_seconds += elapsed_seconds(reduce_start, Clock::now());
        }
        return;
    }
#endif

    auto local_start = Clock::now();
    MatrixReal v_ph = calculate_v_ph_matrix(integrals, ph_basis);
    if (root_rank(settings)) {
        std::cout << "Shape of V_ph matrix: (" << v_ph.rows() << ", " << v_ph.cols() << ")\n";
    }
    MatrixComplex inv_v = linalg_backend.inverse(to_complex(v_ph));
    timings.build_inv_v_seconds = elapsed_seconds(local_start, Clock::now());

    workspace::HostScreeningWorkspace screening(std::move(v_ph), std::move(inv_v), linalg_backend);
    if (settings.execution.frequency_parallel_mode == FrequencyParallelMode::OpenMP) {
        compute_sigma_c_openmp_frequency(orbitals, ph_basis, screening, pq_ph_view, omega_im, weights,
                                         states, settings, sigma_c_im_points, timings);
    } else {
        compute_sigma_c_serial_or_mpi(orbitals, ph_basis, screening, pq_ph_view, omega_im, weights,
                                      states, settings, sigma_c_im_points, timings);
    }

    if (settings.execution.frequency_parallel_mode == FrequencyParallelMode::MPI && settings.execution.mpi_size > 1) {
        auto reduce_start = Clock::now();
        mpi_allreduce_sum_in_place(sigma_c_im_points.data());
        timings.mpi_reduce_seconds = elapsed_seconds(reduce_start, Clock::now());
    }
}

} // namespace gw
