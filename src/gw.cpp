#include "gw/gw.hpp"
#include "gw/frequency_grids.hpp"
#include "gw/mpi_context.hpp"
#include "gw/pade.hpp"
#include "gw/workspace/screening_workspace.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>
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
                               const Tensor3Real& pq_ph,
                               const std::vector<Complex>& omega_im,
                               const std::vector<double>& weights,
                               const std::vector<std::size_t>& states,
                               const GwSettings& settings,
                               MatrixComplex& sigma_c_im_points,
                               GwTimings& local_timings) {
    const Complex omega_n_im = omega_im[f_n];
    std::vector<Complex> current_sigma(orbitals.nmo(), Complex{0.0, 0.0});
    std::vector<double> pk_vec(ph_basis.size(), 0.0);

    for (std::size_t f_prime = 0; f_prime < settings.num_freq_points_total; ++f_prime) {
        const Complex omega_prime_im = omega_im[f_prime];

        auto phase_start = Clock::now();
        const auto pi0_diag = calculate_pi0_ph_diag(omega_prime_im, ph_basis, settings.eta);
        local_timings.build_pi0_seconds += elapsed_seconds(phase_start, Clock::now());

        phase_start = Clock::now();
        const MatrixComplex w_c_ph = screening.compute_w_c(pi0_diag);
        local_timings.invert_epsilon_seconds += elapsed_seconds(phase_start, Clock::now());

        phase_start = Clock::now();
        for (const auto p_idx : states) {
            for (std::size_t k_idx = 0; k_idx < orbitals.nmo(); ++k_idx) {
                for (std::size_t ph = 0; ph < ph_basis.size(); ++ph) {
                    pk_vec[ph] = pq_ph(p_idx, k_idx, ph);
                }

                const Complex w_minus_v = screening.backend().quadratic_form(pk_vec, w_c_ph);

                const Complex g0_denominator = omega_n_im + orbitals.fermi_energy() - orbitals.energy(k_idx);
                const Complex g0_term = g0_denominator /
                    (g0_denominator * g0_denominator - omega_prime_im * omega_prime_im);
                current_sigma[p_idx] -= g0_term * w_minus_v * weights[f_prime];
            }
        }
        local_timings.sigma_c_seconds += elapsed_seconds(phase_start, Clock::now());
    }

    for (const auto p : states) {
        sigma_c_im_points(p, f_n) = current_sigma[p] / kPi;
    }
}

void compute_sigma_c_serial_or_mpi(const OrbitalSpace& orbitals,
                                   const ParticleHoleBasis& ph_basis,
                                   const workspace::HostScreeningWorkspace& screening,
                                   const Tensor3Real& pq_ph,
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
                                  pq_ph,
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
                                      const Tensor3Real& pq_ph,
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
                                      pq_ph,
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
    (void)pq_ph;
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

	// calculate Coulomb on ph-ph space
    start = Clock::now();
    MatrixReal v_ph = calculate_v_ph_matrix(integrals, ph_basis);

	// calculate Coulomb on p-p-ph space
    const Tensor3Real pq_ph = calculate_pq_ph_matrix(integrals, ph_basis);

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
        std::cout << "Shape of V_ph matrix: (" << v_ph.rows() << ", " << v_ph.cols() << ")\n";
        std::cout << "Shape of pq_ph tensor: (" << pq_ph.dim0() << ", " << pq_ph.dim1() << ", " << pq_ph.dim2() << ")\n";
    }

    start = Clock::now();
    MatrixComplex inv_v = linalg_backend.inverse(to_complex(v_ph));
    result.timings.build_inv_v_seconds = elapsed_seconds(start, Clock::now());

    workspace::HostScreeningWorkspace screening(std::move(v_ph), std::move(inv_v), linalg_backend);

    result.sigma_c_im_points = MatrixComplex(orbitals.nmo(), settings.num_freq_points_total, Complex{0.0, 0.0});
    const auto states = selected_states(orbitals.nmo(), settings.selected_state_0based);

	// calculate Sigma_c
    if (root_rank(settings)) {
        std::cout << "\n--- Starting Sigma_c(iw) calculation... ---\n";
    }
    start = Clock::now();
    if (settings.execution.frequency_parallel_mode == FrequencyParallelMode::OpenMP) {
        compute_sigma_c_openmp_frequency(orbitals, ph_basis, screening, pq_ph, omega_im, weights,
                                         states, settings, result.sigma_c_im_points, result.timings);
    } else {
        compute_sigma_c_serial_or_mpi(orbitals, ph_basis, screening, pq_ph, omega_im, weights,
                                      states, settings, result.sigma_c_im_points, result.timings);
    }
    result.timings.sigma_c_wall_seconds = elapsed_seconds(start, Clock::now());

    if (settings.execution.frequency_parallel_mode == FrequencyParallelMode::MPI && settings.execution.mpi_size > 1) {
        start = Clock::now();
        mpi_allreduce_sum_in_place(result.sigma_c_im_points.data());
        result.timings.mpi_reduce_seconds = elapsed_seconds(start, Clock::now());
    }

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
