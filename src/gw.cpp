#include "gw/gw.hpp"
#include "gw/frequency_grids.hpp"
#include "gw/mapping.hpp"
#include "gw/pade.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace gw {
namespace {

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

} // namespace

MatrixReal calculate_exchange(const Tensor4Real& eri_mo, std::size_t nocc) {
    const std::size_t nmo = eri_mo.dim0();
    MatrixReal sigma_x(nmo, nmo, 0.0);
    for (std::size_t q = 0; q < nmo; ++q) {
        for (std::size_t p = 0; p < nmo; ++p) {
            for (std::size_t k = 0; k < nocc; ++k) {
                sigma_x(p, q) -= eri_mo(p, k, q, k);
            }
        }
    }
    return sigma_x;
}

std::vector<Complex> calculate_pi0_ph_diag(Complex omega, const std::vector<double>& mo_energy, std::size_t nocc, std::size_t nvirt, double eta) {
    const std::size_t n_ph = nocc * nvirt;
    std::vector<Complex> diag(n_ph);
    const Complex ieta{0.0, eta};
    for (std::size_t idx = 0; idx < n_ph; ++idx) {
        const auto [i, a] = idx_to_ph(idx, nocc, nvirt);
        const double energy_diff = mo_energy[a] - mo_energy[i];
        const Complex term = Complex{1.0, 0.0} / (omega - energy_diff + ieta)
                           - Complex{1.0, 0.0} / (omega + energy_diff - ieta);
        diag[idx] = 2.0 * term;
    }
    return diag;
}

MatrixReal calculate_v_ph_matrix(const Tensor4Real& eri_mo, std::size_t nocc, std::size_t nvirt) {
    const std::size_t n_ph = nocc * nvirt;
    MatrixReal v_ph(n_ph, n_ph, 0.0);
    for (std::size_t idx_ia = 0; idx_ia < n_ph; ++idx_ia) {
        const auto [i, a] = idx_to_ph(idx_ia, nocc, nvirt);
        for (std::size_t idx_jb = 0; idx_jb < n_ph; ++idx_jb) {
            const auto [j, b] = idx_to_ph(idx_jb, nocc, nvirt);
            v_ph(idx_jb, idx_ia) = eri_mo(j, b, i, a);
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

Tensor3Real calculate_pq_ph_matrix(const Tensor4Real& eri_mo, std::size_t nocc, std::size_t nvirt) {
    const std::size_t nmo = eri_mo.dim0();
    const std::size_t n_ph = nocc * nvirt;
    Tensor3Real out(nmo, nmo, n_ph, 0.0);
    for (std::size_t idx = 0; idx < n_ph; ++idx) {
        const auto [i_ph, a_ph] = idx_to_ph(idx, nocc, nvirt);
        for (std::size_t q = 0; q < nmo; ++q) {
            for (std::size_t p = 0; p < nmo; ++p) {
                out(p, q, idx) = eri_mo(p, q, i_ph, a_ph);
            }
        }
    }
    return out;
}

GwResult run_g0w0(const GwInput& input, const GwSettings& settings) {
    const std::size_t nmo = input.mo_energy.size();
    const std::size_t nocc = input.nocc;
    const std::size_t nvirt = nmo - nocc;
    const std::size_t n_ph = nocc * nvirt;
    if (input.vxc_mo.rows() != nmo || input.vxc_mo.cols() != nmo) {
        throw std::runtime_error("vxc_mo shape is inconsistent with mo_energy");
    }

    auto [omegas, weights] = generate_transformed_legendre_grid(settings.num_freq_points_total);
    std::cout << "Total number of transformed Gauss-Legendre frequency points: " << settings.num_freq_points_total << '\n';

    GwResult result;
    result.omegas = omegas;
    result.weights = weights;
    result.sigma_x = calculate_exchange(input.eri_mo, nocc);

    const MatrixReal v_ph = calculate_v_ph_matrix(input.eri_mo, nocc, nvirt);
    std::cout << "Shape of V_ph matrix: (" << v_ph.rows() << ", " << v_ph.cols() << ")\n";
    const Tensor3Real pq_ph = calculate_pq_ph_matrix(input.eri_mo, nocc, nvirt);
    std::cout << "Shape of pq_ph tensor: (" << pq_ph.dim0() << ", " << pq_ph.dim1() << ", " << pq_ph.dim2() << ")\n";

    std::vector<Complex> omega_im(settings.num_freq_points_total);
    for (std::size_t i = 0; i < settings.num_freq_points_total; ++i) {
        omega_im[i] = Complex{0.0, omegas[i]};
    }

    result.sigma_c_im_points = MatrixComplex(nmo, settings.num_freq_points_total, Complex{0.0, 0.0});
    const auto states = selected_states(nmo, settings.selected_state_0based);

    std::cout << "Starting Sigma_c(iw) calculation...\n";
    for (std::size_t f_n = 0; f_n < settings.num_freq_points_total; ++f_n) {
        const Complex omega_n_im = omega_im[f_n];
        std::vector<Complex> current_sigma(nmo, Complex{0.0, 0.0});

        for (std::size_t f_prime = 0; f_prime < settings.num_freq_points_total; ++f_prime) {
            const Complex omega_prime_im = omega_im[f_prime];
            const auto pi0_diag = calculate_pi0_ph_diag(omega_prime_im, input.mo_energy, nocc, nvirt, settings.eta);
            const MatrixComplex w_c_ph = calculate_w_0_c_matrix(omega_prime_im, v_ph, pi0_diag);

            for (const auto p_idx : states) {
                for (std::size_t k_idx = 0; k_idx < nmo; ++k_idx) {
                    std::vector<double> pk_vec(n_ph, 0.0);
                    for (std::size_t ph = 0; ph < n_ph; ++ph) {
                        pk_vec[ph] = pq_ph(p_idx, k_idx, ph);
                    }

                    std::vector<Complex> tmp(n_ph, Complex{0.0, 0.0});
                    for (std::size_t i = 0; i < n_ph; ++i) {
                        for (std::size_t j = 0; j < n_ph; ++j) {
                            tmp[i] += w_c_ph(i, j) * pk_vec[j];
                        }
                    }
                    Complex w_minus_v{0.0, 0.0};
                    for (std::size_t i = 0; i < n_ph; ++i) {
                        w_minus_v += pk_vec[i] * tmp[i];
                    }

                    const Complex g0_denominator = omega_n_im + input.fermi_energy - input.mo_energy[k_idx];
                    const Complex g0_term = g0_denominator / (g0_denominator * g0_denominator - omega_prime_im * omega_prime_im);
                    current_sigma[p_idx] -= g0_term * w_minus_v * weights[f_prime];
                }
            }
        }
        for (std::size_t p = 0; p < nmo; ++p) {
            result.sigma_c_im_points(p, f_n) = current_sigma[p] / kPi;
        }
        if ((f_n + 1) % 10 == 0 || f_n + 1 == settings.num_freq_points_total) {
            std::cout << "  completed frequency " << (f_n + 1) << " / " << settings.num_freq_points_total << '\n';
        }
    }
    std::cout << "Sigma_c(iw) calculation complete.\n";

    result.qp_energy.assign(nmo, 0.0);
    constexpr std::size_t max_iterations = 200;
    constexpr double tolerance = 1e-6;
    for (std::size_t i = 0; i < nmo; ++i) {
        double current_qp = input.mo_energy[i];
        std::vector<Complex> orbital_sigma(settings.num_freq_points_total);
        for (std::size_t f = 0; f < settings.num_freq_points_total; ++f) {
            orbital_sigma[f] = result.sigma_c_im_points(i, f);
        }
        const auto [coeffs, sampled] = get_pade_coefficients_continued_fraction(omega_im, orbital_sigma, settings.num_pade_params);
        for (std::size_t iter = 0; iter < max_iterations; ++iter) {
            const double target_omega_qp = current_qp - input.fermi_energy;
            const Complex sigma_c_at_qp = evaluate_pade_continued_fraction(coeffs, target_omega_qp, sampled);
            const double new_qp = input.mo_energy[i] + sigma_c_at_qp.real() + result.sigma_x(i, i) - input.vxc_mo(i, i);
            if (std::abs(new_qp - current_qp) < tolerance) {
                current_qp = new_qp;
                break;
            }
            current_qp = new_qp;
        }
        result.qp_energy[i] = current_qp;
    }

    return result;
}

} // namespace gw
