#include "gw/pade.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace gw {

std::pair<std::vector<Complex>, std::vector<Complex>> get_pade_coefficients_continued_fraction(
    const std::vector<Complex>& x,
    const std::vector<Complex>& y,
    std::size_t npar) {
    if (x.size() != y.size()) {
        throw std::runtime_error("Pade: x and y size mismatch");
    }
    if (npar == 0) {
        throw std::runtime_error("Pade: npar must be positive");
    }
    const std::size_t n = x.size();
    if (n == 0) {
        throw std::runtime_error("Pade: no input points");
    }
    const std::size_t npar_actual = std::min(npar, n);
    if (npar_actual == 1) {
        return {{y.front()}, {x.front()}};
    }

    std::vector<Complex> sampled_x(npar_actual);
    std::vector<Complex> sampled_y(npar_actual);
    const std::size_t n_step = n / (npar_actual - 1);
    if (n_step == 0) {
        for (std::size_t i = 0; i < npar_actual; ++i) {
            sampled_x[i] = x[i];
            sampled_y[i] = y[i];
        }
    } else {
        std::size_t i_dat = 0;
        for (std::size_t i_par = 0; i_par < npar_actual - 1; ++i_par) {
            sampled_x[i_par] = x[i_dat];
            sampled_y[i_par] = y[i_dat];
            i_dat += n_step;
        }
        sampled_x[npar_actual - 1] = x[n - 1];
        sampled_y[npar_actual - 1] = y[n - 1];
    }

    MatrixComplex g(npar_actual, npar_actual, Complex{0.0, 0.0});
    for (std::size_t i = 0; i < npar_actual; ++i) {
        g(i, 0) = sampled_y[i];
    }

    for (std::size_t i_par = 1; i_par < npar_actual; ++i_par) {
        for (std::size_t i_dat = i_par; i_dat < npar_actual; ++i_dat) {
            const Complex numerator = g(i_par - 1, i_par - 1) - g(i_dat, i_par - 1);
            const Complex denominator = (sampled_x[i_dat] - sampled_x[i_par - 1]) * g(i_dat, i_par - 1);
            if (std::abs(denominator) < 1e-20) {
                g(i_dat, i_par) = numerator / Complex{1e-20, 0.0};
            } else {
                g(i_dat, i_par) = numerator / denominator;
            }
        }
    }

    std::vector<Complex> coeffs(npar_actual);
    for (std::size_t i = 0; i < npar_actual; ++i) {
        coeffs[i] = g(i, i);
    }
    return {coeffs, sampled_x};
}

Complex evaluate_pade_continued_fraction(
    const std::vector<Complex>& coeffs,
    double z_target_real,
    const std::vector<Complex>& x_points_for_pade) {
    const std::size_t npar = coeffs.size();
    if (npar == 0) {
        return {0.0, 0.0};
    }
    if (npar == 1) {
        return coeffs.front();
    }
    if (x_points_for_pade.size() < npar - 1) {
        throw std::runtime_error("Pade evaluation: insufficient sampled x points");
    }

    const Complex z_target{z_target_real, 0.0};
    Complex val{1.0, 0.0};
    for (std::size_t k_rev = npar - 1; k_rev > 0; --k_rev) {
        const std::size_t k = k_rev - 1;
        const Complex numerator = coeffs[k + 1] * (z_target - x_points_for_pade[k]);
        if (std::abs(val) < 1e-20) {
            val = {1e-20, 0.0};
        }
        val = Complex{1.0, 0.0} + numerator / val;
    }
    if (std::abs(val) < 1e-20) {
        return coeffs.front() / Complex{1e-20, 0.0};
    }
    return coeffs.front() / val;
}

} // namespace gw
