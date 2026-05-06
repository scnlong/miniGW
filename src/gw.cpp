#include "gw/gw.hpp"
#include "gw/frequency_grids.hpp"
#include "gw/pade.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace gw {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

MatrixComplex identity_complex(std::size_t n) {
    MatrixComplex out(n, n, Complex{0.0, 0.0});
    for (std::size_t i = 0; i < n; ++i) {
        out(i, i) = Complex{1.0, 0.0};
    }
    return out;
}

MatrixComplex to_complex(const MatrixReal& in) {
    MatrixComplex out(in.rows(), in.cols(), Complex{0.0, 0.0});
    for (std::size_t i = 0; i < in.rows(); ++i) {
        for (std::size_t j = 0; j < in.cols(); ++j) {
            out(i, j) = Complex{in(i, j), 0.0};
        }
    }
    return out;
}

MatrixComplex inverse(MatrixComplex a) {
    const std::size_t n = a.rows();
    if (a.rows() != a.cols()) {
        throw std::runtime_error("inverse: matrix must be square");
    }
    MatrixComplex inv = identity_complex(n);

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        double pivot_abs = std::abs(a(col, col));
        for (std::size_t row = col + 1; row < n; ++row) {
            const double candidate = std::abs(a(row, col));
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot = row;
            }
        }
        if (pivot_abs < 1e-14) {
            throw std::runtime_error("inverse: near-singular matrix");
        }
        if (pivot != col) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(a(col, j), a(pivot, j));
                std::swap(inv(col, j), inv(pivot, j));
            }
        }

        const Complex diag = a(col, col);
        for (std::size_t j = 0; j < n; ++j) {
            a(col, j) /= diag;
            inv(col, j) /= diag;
        }
        for (std::size_t row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const Complex factor = a(row, col);
            if (std::abs(factor) == 0.0) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                a(row, j) -= factor * a(col, j);
                inv(row, j) -= factor * inv(col, j);
            }
        }
    }
    return inv;
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

} // namespace

MatrixReal calculate_exchange(const OrbitalSpace& orbitals, const MolecularIntegrals& integrals) {
    validate_input_shapes(orbitals, integrals);

    MatrixReal sigma_x(orbitals.nmo(), orbitals.nmo(), 0.0);
    for (std::size_t q = 0; q < orbitals.nmo(); ++q) {
        for (std::size_t p = 0; p < orbitals.nmo(); ++p) {
            for (std::size_t k_occ = 0; k_occ < orbitals.nocc(); ++k_occ) {
                sigma_x(p, q) -= integrals.eri(p, k_occ, q, k_occ);
            }
        }
    }
    return sigma_x;
}

std::vector<Complex> calculate_pi0_ph_diag(Complex omega, const ParticleHoleBasis& ph_basis, double eta) {
    std::vector<Complex> diag(ph_basis.size());
    const Complex ieta{0.0, eta};
    for (std::size_t ph = 0; ph < ph_basis.size(); ++ph) {
        const ParticleHolePair& ia = ph_basis[ph];
        const Complex term = Complex{1.0, 0.0} / (omega - ia.delta_e + ieta)
                           - Complex{1.0, 0.0} / (omega + ia.delta_e - ieta);
        diag[ph] = 2.0 * term;
    }
    return diag;
}

MatrixReal calculate_v_ph_matrix(const MolecularIntegrals& integrals, const ParticleHoleBasis& ph_basis) {
    MatrixReal v_ph(ph_basis.size(), ph_basis.size(), 0.0);
    for (std::size_t ph_ia = 0; ph_ia < ph_basis.size(); ++ph_ia) {
        const ParticleHolePair& ia = ph_basis[ph_ia];
        for (std::size_t ph_jb = 0; ph_jb < ph_basis.size(); ++ph_jb) {
            const ParticleHolePair& jb = ph_basis[ph_jb];
            // Preserve the original Julia/C++ prototype ordering:
            // v_ph[jb, ia] = eri(j, b, i, a).
            v_ph(ph_jb, ph_ia) = integrals.eri(jb.i_occ, jb.a_mo, ia.i_occ, ia.a_mo);
        }
    }
    return v_ph;
}

MatrixComplex calculate_w_0_c_matrix(Complex /*omega*/, const MatrixReal& v_ph, const std::vector<Complex>& pi0_diag) {
    const std::size_t n = v_ph.rows();
    if (v_ph.rows() != v_ph.cols() || pi0_diag.size() != n) {
        throw std::runtime_error("calculate_w_0_c_matrix: inconsistent dimensions");
    }

    MatrixComplex epsilon(n, n, Complex{0.0, 0.0});
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            epsilon(i, k) = -Complex{v_ph(i, k), 0.0} * pi0_diag[k];
        }
        epsilon(i, i) += Complex{1.0, 0.0};
    }

    MatrixComplex inv_eps = inverse(epsilon);
    for (std::size_t i = 0; i < n; ++i) {
        inv_eps(i, i) -= Complex{1.0, 0.0};
    }
    const MatrixComplex inv_v = inverse(to_complex(v_ph));

    // Julia prototype: @tensor W[i,k] := tmp1[j,i] * tmp2[j,k]
    MatrixComplex w(n, n, Complex{0.0, 0.0});
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            Complex sum{0.0, 0.0};
            for (std::size_t j = 0; j < n; ++j) {
                sum += inv_eps(j, i) * inv_v(j, k);
            }
            w(i, k) = sum;
        }
    }
    return w;
}

Tensor3Real calculate_pq_ph_matrix(const MolecularIntegrals& integrals, const ParticleHoleBasis& ph_basis) {
    const std::size_t nmo = integrals.nmo();
    Tensor3Real out(nmo, nmo, ph_basis.size(), 0.0);
    for (std::size_t ph = 0; ph < ph_basis.size(); ++ph) {
        const ParticleHolePair& ia = ph_basis[ph];
        for (std::size_t q = 0; q < nmo; ++q) {
            for (std::size_t p = 0; p < nmo; ++p) {
                out(p, q, ph) = integrals.eri(p, q, ia.i_occ, ia.a_mo);
            }
        }
    }
    return out;
}

GwResult run_g0w0(const GwInput& input, const GwSettings& settings) {
    OrbitalSpace orbitals(input.mo_energy, input.nocc, input.fermi_energy);
    MolecularIntegrals integrals(input.eri_mo, input.vxc_mo);
    validate_input_shapes(orbitals, integrals);

    const ParticleHoleBasis ph_basis(orbitals);

    auto [omegas, weights] = generate_transformed_legendre_grid(settings.num_freq_points_total);
	std::cout << "Total number of Padé  parameters: " << settings.num_pade_params << '\n';
    std::cout << "Total number of transformed Gauss-Legendre frequency points: " << settings.num_freq_points_total << '\n';

    GwResult result;
    result.omegas = omegas;
    result.weights = weights;

    auto start = Clock::now();
    result.sigma_x = calculate_exchange(orbitals, integrals);
    result.timings.exchange_seconds = elapsed_seconds(start, Clock::now());

    start = Clock::now();
    const MatrixReal v_ph = calculate_v_ph_matrix(integrals, ph_basis);
    std::cout << "Shape of V_ph matrix: (" << v_ph.rows() << ", " << v_ph.cols() << ")\n";
    const Tensor3Real pq_ph = calculate_pq_ph_matrix(integrals, ph_basis);
    std::cout << "Shape of pq_ph tensor: (" << pq_ph.dim0() << ", " << pq_ph.dim1() << ", " << pq_ph.dim2() << ")\n";

    std::vector<Complex> omega_im(settings.num_freq_points_total);
    for (std::size_t i = 0; i < settings.num_freq_points_total; ++i) {
        omega_im[i] = Complex{0.0, omegas[i]};
    }
    result.timings.build_mapping_seconds = elapsed_seconds(start, Clock::now());

    result.sigma_c_im_points = MatrixComplex(orbitals.nmo(), settings.num_freq_points_total, Complex{0.0, 0.0});
    const auto states = selected_states(orbitals.nmo(), settings.selected_state_0based);

    std::cout << "Starting Sigma_c(iw) calculation...\n";
    start = Clock::now();
    auto frequency_chunk_start = start;
    for (std::size_t f_n = 0; f_n < settings.num_freq_points_total; ++f_n) {
        const Complex omega_n_im = omega_im[f_n];
        std::vector<Complex> current_sigma(orbitals.nmo(), Complex{0.0, 0.0});

        for (std::size_t f_prime = 0; f_prime < settings.num_freq_points_total; ++f_prime) {
            const Complex omega_prime_im = omega_im[f_prime];
            auto phase_start = Clock::now();
            const auto pi0_diag = calculate_pi0_ph_diag(omega_prime_im, ph_basis, settings.eta);
            result.timings.build_pi0_seconds += elapsed_seconds(phase_start, Clock::now());

            phase_start = Clock::now();
            const MatrixComplex w_c_ph = calculate_w_0_c_matrix(omega_prime_im, v_ph, pi0_diag);
            result.timings.invert_epsilon_seconds += elapsed_seconds(phase_start, Clock::now());

            phase_start = Clock::now();
            for (const auto p_idx : states) {
                for (std::size_t k_idx = 0; k_idx < orbitals.nmo(); ++k_idx) {
                    std::vector<double> pk_vec(ph_basis.size(), 0.0);
                    for (std::size_t ph = 0; ph < ph_basis.size(); ++ph) {
                        pk_vec[ph] = pq_ph(p_idx, k_idx, ph);
                    }

                    std::vector<Complex> tmp(ph_basis.size(), Complex{0.0, 0.0});
                    for (std::size_t i = 0; i < ph_basis.size(); ++i) {
                        for (std::size_t j = 0; j < ph_basis.size(); ++j) {
                            tmp[i] += w_c_ph(i, j) * pk_vec[j];
                        }
                    }
                    Complex w_minus_v{0.0, 0.0};
                    for (std::size_t i = 0; i < ph_basis.size(); ++i) {
                        w_minus_v += pk_vec[i] * tmp[i];
                    }

                    const Complex g0_denominator = omega_n_im + orbitals.fermi_energy() - orbitals.energy(k_idx);
                    const Complex g0_term = g0_denominator / (g0_denominator * g0_denominator - omega_prime_im * omega_prime_im);
                    current_sigma[p_idx] -= g0_term * w_minus_v * weights[f_prime];
                }
            }
            result.timings.sigma_c_seconds += elapsed_seconds(phase_start, Clock::now());
        }
        for (std::size_t p = 0; p < orbitals.nmo(); ++p) {
            result.sigma_c_im_points(p, f_n) = current_sigma[p] / kPi;
        }
        if ((f_n + 1) % 10 == 0 || f_n + 1 == settings.num_freq_points_total) {
            const auto frequency_chunk_end = Clock::now();
            std::cout << "  completed frequency " << (f_n + 1) << " / " << settings.num_freq_points_total
                      << " in " << elapsed_seconds(frequency_chunk_start, frequency_chunk_end) << " s\n";
            frequency_chunk_start = frequency_chunk_end;
        }
    }
    std::cout << "Sigma_c(iw) calculation complete.\n";
    result.timings.sigma_c_seconds = elapsed_seconds(start, Clock::now())
                                   - result.timings.build_pi0_seconds
                                   - result.timings.invert_epsilon_seconds;

    start = Clock::now();
    result.qp_energy.assign(orbitals.nmo(), 0.0);
    constexpr std::size_t max_iterations = 200;
    constexpr double tolerance = 1e-6;
    for (std::size_t state = 0; state < orbitals.nmo(); ++state) {
        double current_qp = orbitals.energy(state);
        std::vector<Complex> orbital_sigma(settings.num_freq_points_total);
        for (std::size_t f = 0; f < settings.num_freq_points_total; ++f) {
            orbital_sigma[f] = result.sigma_c_im_points(state, f);
        }
        const auto [coeffs, sampled] = get_pade_coefficients_continued_fraction(omega_im, orbital_sigma, settings.num_pade_params);
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

    return result;
}

} // namespace gw
