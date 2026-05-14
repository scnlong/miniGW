#include "gw/gw.hpp"
#include "gw/frequency_grids.hpp"
#include "gw/mpi_context.hpp"
#include "gw/pade.hpp"
#include "gw/workspace/screening_workspace.hpp"
#include "gw/workspace/pq_ph_panel.hpp"
#ifdef GW_HAS_CUDA_BACKEND
#include "gw/linalg_cublas.hpp"
#include "gw/workspace/device_screening_workspace.hpp"
#include "gw/workspace/device_pq_ph_panel.hpp"
#endif
#ifdef GW_HAS_HIP_BACKEND
#include "gw/linalg_hip.hpp"
#include "gw/workspace/hip_screening_workspace.hpp"
#include "gw/workspace/hip_pq_ph_panel.hpp"
#endif
#ifdef GW_HAS_SCALAPACK_BACKEND
#include "gw/workspace/distributed_screening_workspace.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

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

#ifdef GW_HAS_HIP_BACKEND
void compute_sigma_c_frequency_hip(std::size_t f_n,
                                      const OrbitalSpace& orbitals,
                                      const ParticleHoleBasis& ph_basis,
                                      workspace::HipScreeningWorkspace& screening,
                                      workspace::HipPqPhPanelView& pq_ph_view,
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
                const workspace::HipComplexPanelView pk_panel = pq_ph_view.fill_panel(p_idx, k0, width);

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

    for (const auto p : states) {
        sigma_c_im_points(p, f_n) = current_sigma[p] / kPi;
    }
}

void compute_sigma_c_hip_screening(const OrbitalSpace& orbitals,
                                      const ParticleHoleBasis& ph_basis,
                                      workspace::HipScreeningWorkspace& screening,
                                      workspace::HipPqPhPanelView& pq_ph_view,
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
        compute_sigma_c_frequency_hip(f_n,
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
            std::cout << "  completed HIP/ROCm device-resident frequency " << (f_n + 1) << " / "
                      << nfreq << " in " << elapsed_seconds(chunk_start, now) << " s\n";
            chunk_start = now;
        } else if (mpi_frequency && (local_completed == local_total || local_completed % 10U == 0U)) {
            const auto now = Clock::now();
            std::cout << "  rank " << settings.execution.mpi_rank
                      << " completed " << local_completed << " / " << local_total
                      << " assigned HIP/ROCm frequencies on device " << settings.execution.hip_device_id
                      << " (last global frequency " << (f_n + 1) << " / " << nfreq << ") in "
                      << elapsed_seconds(chunk_start, now) << " s\n";
            chunk_start = now;
        }
    }
}
#endif

#ifdef GW_HAS_SCALAPACK_BACKEND
void compute_sigma_c_frequency_distributed(std::size_t f_n,
                                           const OrbitalSpace& orbitals,
                                           const ParticleHoleBasis& ph_basis,
                                           workspace::DistributedScreeningWorkspace& screening,
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

    // For grouped ScaLAPACK frequency parallelism, every rank in the same
    // ScaLAPACK communicator group obtains the same scalar panel contractions
    // after group-local reductions.  Only the group root must contribute the
    // completed frequency column to the world-level reduction; otherwise the
    // final MPI_Allreduce over MPI_COMM_WORLD would count the same group result
    // once per rank in the group.
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

void compute_sigma_c_distributed_screening(const OrbitalSpace& orbitals,
                                           const ParticleHoleBasis& ph_basis,
                                           workspace::DistributedScreeningWorkspace& screening,
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

// exchange
MatrixReal calculate_exchange(const OrbitalSpace& orbitals, const MolecularIntegrals& integrals) {
    validate_input_shapes(orbitals, integrals);

    MatrixReal sigma_x(orbitals.nmo(), orbitals.nmo(), 0.0);
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for collapse(2) schedule(static) if(kernel_loops_enabled())
#endif
    for (std::size_t q = 0; q < orbitals.nmo(); ++q) {
        for (std::size_t p = 0; p < orbitals.nmo(); ++p) {
            double value = 0.0;
            for (std::size_t k_occ = 0; k_occ < orbitals.nocc(); ++k_occ) {
                value -= integrals.eri(p, k_occ, q, k_occ);
            }
            sigma_x(p, q) = value;
        }
    }
    return sigma_x;
}

// Polarisability
std::vector<Complex> calculate_pi0_ph_diag(Complex omega, const ParticleHoleBasis& ph_basis, double eta) {
    std::vector<Complex> diag(ph_basis.size());
    const Complex ieta{0.0, eta};
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for schedule(static) if(kernel_loops_enabled())
#endif
    for (std::size_t ph = 0; ph < ph_basis.size(); ++ph) {
        const ParticleHolePair& ia = ph_basis[ph];
        const Complex term = Complex{1.0, 0.0} / (omega - ia.delta_e + ieta)
                           - Complex{1.0, 0.0} / (omega + ia.delta_e - ieta);
        diag[ph] = 2.0 * term;
    }
    return diag;
}

// Coulomb on ph-ph space
MatrixReal calculate_v_ph_matrix(const MolecularIntegrals& integrals, const ParticleHoleBasis& ph_basis) {
    MatrixReal v_ph(ph_basis.size(), ph_basis.size(), 0.0);
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for collapse(2) schedule(static) if(kernel_loops_enabled())
#endif
    for (std::size_t ph_ia = 0; ph_ia < ph_basis.size(); ++ph_ia) {
        for (std::size_t ph_jb = 0; ph_jb < ph_basis.size(); ++ph_jb) {
            const ParticleHolePair& ia = ph_basis[ph_ia];
            const ParticleHolePair& jb = ph_basis[ph_jb];
            // Convention used by epsilon: epsilon[row, col] = delta[row, col] - V_ph[row, col] * Pi0[col].
            // Therefore row = jb, col = ia, and V_ph[jb, ia] = (j b | i a).
            v_ph(ph_jb, ph_ia) = integrals.eri(jb.i_occ, jb.a_mo, ia.i_occ, ia.a_mo);
        }
    }
    return v_ph;
}

// W term
MatrixComplex calculate_w_0_c_matrix(const MatrixReal& v_ph,
                                     const MatrixComplex& inv_v,
                                     const std::vector<Complex>& pi0_diag,
                                     const linalg::Backend& backend) {
    const std::size_t n = v_ph.rows();
    if (v_ph.rows() != v_ph.cols() || pi0_diag.size() != n || inv_v.rows() != n || inv_v.cols() != n) {
        throw std::runtime_error("calculate_w_0_c_matrix: inconsistent dimensions");
    }

    MatrixComplex epsilon(n, n, Complex{0.0, 0.0});
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for collapse(2) schedule(static) if(kernel_loops_enabled())
#endif
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            epsilon(i, k) = -Complex{v_ph(i, k), 0.0} * pi0_diag[k];
        }
    }
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for schedule(static) if(kernel_loops_enabled())
#endif
    for (std::size_t i = 0; i < n; ++i) {
        epsilon(i, i) += Complex{1.0, 0.0};
    }

    MatrixComplex inv_eps = backend.inverse(epsilon);
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for schedule(static) if(kernel_loops_enabled())
#endif
    for (std::size_t i = 0; i < n; ++i) {
        inv_eps(i, i) -= Complex{1.0, 0.0};
    }

    // This is W = inv_eps^T * inv_v; 
    return backend.gemm(inv_eps, inv_v, linalg::MatrixTranspose::Transpose, linalg::MatrixTranspose::NoTranspose);
}

// Coulomb on p-p-ph space
Tensor3Real calculate_pq_ph_matrix(const MolecularIntegrals& integrals, const ParticleHoleBasis& ph_basis) {
    const std::size_t nmo = integrals.nmo();
    Tensor3Real out(nmo, nmo, ph_basis.size(), 0.0);
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for collapse(3) schedule(static) if(kernel_loops_enabled())
#endif
    for (std::size_t ph = 0; ph < ph_basis.size(); ++ph) {
        for (std::size_t q = 0; q < nmo; ++q) {
            for (std::size_t p = 0; p < nmo; ++p) {
                const ParticleHolePair& ia = ph_basis[ph];
                out(p, q, ph) = integrals.eri(p, q, ia.i_occ, ia.a_mo);
            }
        }
    }
    return out;
}

// g0w0 main function
GwResult run_g0w0(const GwInput& input, const GwSettings& settings) {
    const auto run_start = Clock::now();
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
    g_openmp_kernel_loops_enabled.store(settings.execution.openmp_kernel_loops, std::memory_order_relaxed);
    workspace::set_host_screening_openmp_kernel_loops(settings.execution.openmp_kernel_loops);
#endif

	// set up parameters
    OrbitalSpace orbitals(input.mo_energy, input.nocc, input.fermi_energy);
    MolecularIntegrals integrals(input.eri_mo, input.vxc_mo);
    validate_input_shapes(orbitals, integrals);

    const ParticleHoleBasis ph_basis(orbitals);
    const linalg::Backend& linalg_backend = settings.linalg_backend ? *settings.linalg_backend : linalg::reference_backend();
    const auto caps = linalg_backend.capabilities();

    if (settings.execution.frequency_parallel_mode == FrequencyParallelMode::OpenMP) {
        if (!caps.thread_safe) {
            throw std::runtime_error("OpenMP frequency parallelism requires a thread-safe linear algebra backend.");
        }
        if (caps.distributed_mpi || caps.uses_device_memory) {
            throw std::runtime_error("OpenMP frequency parallelism is restricted to local CPU backends. Use MPI frequency distribution or a backend-managed GPU task queue for distributed/device backends.");
        }
        if (caps.uses_internal_threads) {
            if (root_rank(settings)) {
                std::cout << "Warning: OpenMP frequency parallelism with a threaded BLAS/LAPACK backend can oversubscribe CPUs. Prefer OPENBLAS_NUM_THREADS=1.\n";
            }
        }
    }

    if (root_rank(settings)) {
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

    auto [omegas, weights] = generate_transformed_legendre_grid(settings.num_freq_points_total);
    if (root_rank(settings)) {
        std::cout << "Total number of Pade parameters: " << settings.num_pade_params << '\n';
        std::cout << "Total number of transformed Gauss-Legendre frequency points: " << settings.num_freq_points_total << '\n';
    }

    GwResult result;
    result.omegas = omegas;
    result.weights = weights;

	// calculate exchange
    auto start = Clock::now();
    result.sigma_x = calculate_exchange(orbitals, integrals);
    result.timings.exchange_seconds = elapsed_seconds(start, Clock::now());

	// Build a lightweight panel view for P_{pk,ia} = (p k | i a).
    // This removes the resident Tensor3Real(nmo,nmo,n_ph) pq_ph allocation;
    // panels are generated on demand from the current MolecularIntegrals view.
    // ERI itself is still replicated in this version.
    start = Clock::now();
    const workspace::PqPhPanelView pq_ph_view(integrals, ph_basis);

	// set up freq points
    std::vector<Complex> omega_im(settings.num_freq_points_total);
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for schedule(static) if(kernel_loops_enabled())
#endif
    for (std::size_t i = 0; i < settings.num_freq_points_total; ++i) {
        omega_im[i] = Complex{0.0, omegas[i]};
    }
    result.timings.build_mapping_seconds = elapsed_seconds(start, Clock::now());

    if (root_rank(settings)) {
        std::cout << "pq_ph panel view: nmo=" << pq_ph_view.nmo()
                  << ", n_ph=" << pq_ph_view.nph()
                  << ", panel size=" << settings.contraction_panel_size << '\n';
    }

    result.sigma_c_im_points = MatrixComplex(orbitals.nmo(), settings.num_freq_points_total, Complex{0.0, 0.0});
    const auto states = selected_states(orbitals.nmo(), settings.selected_state_0based);

	// calculate Sigma_c
    if (root_rank(settings)) {
        std::cout << "\n--- Starting Sigma_c(iw) calculation... ---\n";
    }
    start = Clock::now();

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
        result.timings.build_inv_v_seconds = elapsed_seconds(device_start, Clock::now());
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
                                         states, cuda_settings, result.sigma_c_im_points, result.timings);
        if (cuda_settings.execution.frequency_parallel_mode == FrequencyParallelMode::MPI &&
            cuda_settings.execution.mpi_size > 1) {
            auto reduce_start = Clock::now();
            mpi_allreduce_sum_in_place(result.sigma_c_im_points.data());
            result.timings.mpi_reduce_seconds += elapsed_seconds(reduce_start, Clock::now());
        }
    } else
#endif
#ifdef GW_HAS_HIP_BACKEND
    if (std::string_view{linalg_backend.name()}.find("hipblas") != std::string_view::npos) {
        if (settings.execution.frequency_parallel_mode == FrequencyParallelMode::OpenMP) {
            throw std::runtime_error("Device-resident HIP/ROCm screening supports serial or MPI frequency parallelism, not OpenMP frequency parallelism.");
        }
        GwSettings hip_settings = settings;
        hip_settings.execution.hip_device_id = linalg::set_hip_device_for_local_rank(
            settings.execution.mpi_local_rank,
            settings.execution.tasks_per_gpu);
        if (root_rank(hip_settings)) {
            std::cout << "Using HIP/ROCm device-resident screening workspace for V_ph/epsilon/W_c.\n";
            std::cout << "ERI is still host-replicated; HIP pq panels use an auto-selected full-resident or streaming panel source.\n";
            std::cout << "HIP/ROCm MPI rank-to-GPU policy: tasks-per-gpu=" << hip_settings.execution.tasks_per_gpu
                      << ", local rank/size=" << hip_settings.execution.mpi_local_rank << " / "
                      << hip_settings.execution.mpi_local_size << ", selected device="
                      << hip_settings.execution.hip_device_id << "\n";
        }
        auto hip_start = Clock::now();
        MatrixReal v_ph = calculate_v_ph_matrix(integrals, ph_basis);
        workspace::HipScreeningWorkspace screening(v_ph);
        workspace::HipPqPhPanelView hip_pq_ph_view(integrals, ph_basis, hip_settings.contraction_panel_size);
        result.timings.build_inv_v_seconds = elapsed_seconds(hip_start, Clock::now());
        if (root_rank(hip_settings)) {
            std::cout << "HIP/ROCm screening workspace estimated device allocation after setup: "
                      << static_cast<double>(screening.estimated_device_bytes()) / (1024.0 * 1024.0)
                      << " MiB\n";
            std::cout << "HIP/ROCm pq-panel source mode: " << hip_pq_ph_view.storage_mode_name() << '\n';
            std::cout << "HIP/ROCm pq-panel source estimated device allocation: "
                      << static_cast<double>(hip_pq_ph_view.estimated_device_bytes()) / (1024.0 * 1024.0)
                      << " MiB\n";
            std::cout << "HIP/ROCm pq-panel source estimated pinned host staging: "
                      << static_cast<double>(hip_pq_ph_view.estimated_host_pinned_bytes()) / (1024.0 * 1024.0)
                      << " MiB\n";
        }
        compute_sigma_c_hip_screening(orbitals, ph_basis, screening, hip_pq_ph_view, omega_im, weights,
                                      states, hip_settings, result.sigma_c_im_points, result.timings);
        if (hip_settings.execution.frequency_parallel_mode == FrequencyParallelMode::MPI &&
            hip_settings.execution.mpi_size > 1) {
            auto reduce_start = Clock::now();
            mpi_allreduce_sum_in_place(result.sigma_c_im_points.data());
            result.timings.mpi_reduce_seconds += elapsed_seconds(reduce_start, Clock::now());
        }
    } else
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
        const auto distributed_gemm_provider = caps.uses_cosma_pxgemm
                                                ? matrix::DistributedGemmProvider::CosmaPrefixedPxgemm
                                                : matrix::DistributedGemmProvider::Scalapack;
        workspace::DistributedScreeningWorkspace screening(integrals,
                                                           ph_basis,
                                                           64,
                                                           settings.execution.frequency_group_size,
                                                           distributed_gemm_provider);
        result.timings.build_inv_v_seconds = elapsed_seconds(distributed_start, Clock::now());
        compute_sigma_c_distributed_screening(orbitals, ph_basis, screening, pq_ph_view, omega_im, weights,
                                              states, settings, result.sigma_c_im_points, result.timings);
        if (settings.execution.frequency_parallel_mode == FrequencyParallelMode::MPI && settings.execution.mpi_size > 1) {
            auto reduce_start = Clock::now();
            mpi_allreduce_sum_in_place(result.sigma_c_im_points.data());
            result.timings.mpi_reduce_seconds += elapsed_seconds(reduce_start, Clock::now());
        }
    } else
#endif
    {
        // Local host path: build replicated V_ph and use the selected local backend
        // for inv(V_ph), epsilon inverse and GEMM.
        auto local_start = Clock::now();
        MatrixReal v_ph = calculate_v_ph_matrix(integrals, ph_basis);
        if (root_rank(settings)) {
            std::cout << "Shape of V_ph matrix: (" << v_ph.rows() << ", " << v_ph.cols() << ")\n";
        }
        MatrixComplex inv_v = linalg_backend.inverse(to_complex(v_ph));
        result.timings.build_inv_v_seconds = elapsed_seconds(local_start, Clock::now());

        workspace::HostScreeningWorkspace screening(std::move(v_ph), std::move(inv_v), linalg_backend);
        if (settings.execution.frequency_parallel_mode == FrequencyParallelMode::OpenMP) {
            compute_sigma_c_openmp_frequency(orbitals, ph_basis, screening, pq_ph_view, omega_im, weights,
                                             states, settings, result.sigma_c_im_points, result.timings);
        } else {
            compute_sigma_c_serial_or_mpi(orbitals, ph_basis, screening, pq_ph_view, omega_im, weights,
                                          states, settings, result.sigma_c_im_points, result.timings);
        }

        if (settings.execution.frequency_parallel_mode == FrequencyParallelMode::MPI && settings.execution.mpi_size > 1) {
            auto reduce_start = Clock::now();
            mpi_allreduce_sum_in_place(result.sigma_c_im_points.data());
            result.timings.mpi_reduce_seconds = elapsed_seconds(reduce_start, Clock::now());
        }
    }
    result.timings.sigma_c_wall_seconds = elapsed_seconds(start, Clock::now());

    if (root_rank(settings)) {
        std::cout << "Sigma_c(iw) calculation complete.\n";
    }

    start = Clock::now();
    result.qp_energy.assign(orbitals.nmo(), 0.0);
    constexpr std::size_t max_iterations = 200;
    constexpr double tolerance = 1e-6;

	// analytical continuation
    for (std::size_t state = 0; state < orbitals.nmo(); ++state) {
        double current_qp = orbitals.energy(state);
        std::vector<Complex> orbital_sigma(settings.num_freq_points_total);
        for (std::size_t f = 0; f < settings.num_freq_points_total; ++f) {
            orbital_sigma[f] = result.sigma_c_im_points(state, f);
        }

		// calculate Pade coefficients
        const auto [coeffs, sampled] = get_pade_coefficients_continued_fraction(omega_im, orbital_sigma, settings.num_pade_params);

		// find qp energy
        for (std::size_t iter = 0; iter < max_iterations; ++iter) {
            const double target_omega_qp = current_qp - orbitals.fermi_energy();
            const Complex sigma_c_at_qp = evaluate_pade_continued_fraction(coeffs, target_omega_qp, sampled);
            const double new_qp = orbitals.energy(state) + sigma_c_at_qp.real() + result.sigma_x(state, state) - integrals.vxc(state, state);
            if (std::abs(new_qp - current_qp) < tolerance) {
                current_qp = new_qp;
                break;
            }
            current_qp = new_qp;
        }
        result.qp_energy[state] = current_qp;
    }
    result.timings.pade_seconds = elapsed_seconds(start, Clock::now());
    result.timings.total_wall_seconds = elapsed_seconds(run_start, Clock::now());

    return result;
}

} // namespace gw
