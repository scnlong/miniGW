#include "gw/frequency_grids.hpp"
#include "gw/types.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace gw {
namespace {

std::pair<std::vector<double>, std::vector<double>> gauleg(double x1, double x2, std::size_t n) {
    std::vector<double> x(n, 0.0);
    std::vector<double> w(n, 0.0);
    constexpr double eps = 3e-14;
    const std::size_t m = (n + 1) / 2;
    const double xm = 0.5 * (x2 + x1);
    const double xl = 0.5 * (x2 - x1);

    for (std::size_t i = 0; i < m; ++i) {
        double z = std::cos(kPi * (static_cast<double>(i + 1) - 0.25) / (static_cast<double>(n) + 0.5));
        double pp = 0.0;
        for (int iter = 0; iter < 100; ++iter) {
            double p1 = 1.0;
            double p2 = 0.0;
            for (std::size_t j = 1; j <= n; ++j) {
                const double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * static_cast<double>(j) - 1.0) * z * p2 - (static_cast<double>(j) - 1.0) * p3) / static_cast<double>(j);
            }
            pp = static_cast<double>(n) * (z * p1 - p2) / (z * z - 1.0);
            const double z_old = z;
            z = z_old - p1 / pp;
            if (std::abs(z - z_old) <= eps) {
                break;
            }
        }
        x[i] = xm - xl * z;
        x[n - 1 - i] = xm + xl * z;
        w[i] = 2.0 * xl / ((1.0 - z * z) * pp * pp);
        w[n - 1 - i] = w[i];
    }
    return {x, w};
}

} // namespace

std::pair<std::vector<double>, std::vector<double>> generate_linear_grid(std::size_t num_freq_points, double max_freq) {
    if (num_freq_points == 0 || max_freq <= 0.0) {
        throw std::runtime_error("generate_linear_grid: invalid arguments");
    }
    std::vector<double> values(num_freq_points, 0.0);
    std::vector<double> weights(num_freq_points, 0.0);
    const double step = max_freq / static_cast<double>(num_freq_points);
    for (std::size_t i = 1; i < num_freq_points; ++i) {
        values[i] = step * static_cast<double>(i + 1); // Matches the Julia/Fortran convention in the prototype.
    }
    std::fill(weights.begin(), weights.end(), values.size() > 1 ? values[1] - values[0] : step);
    return {values, weights};
}

std::pair<std::vector<double>, std::vector<double>> generate_transformed_legendre_grid(std::size_t num_freq_points) {
    if (num_freq_points == 0) {
        throw std::runtime_error("generate_transformed_legendre_grid: num_freq_points must be positive");
    }
    auto [points, weights] = gauleg(-1.0, 1.0, num_freq_points);
    std::vector<double> transformed_values(num_freq_points, 0.0);
    std::vector<double> transformed_weights(num_freq_points, 0.0);

    for (std::size_t i = 0; i < num_freq_points; ++i) {
        const double y = points[i];
        const double wy = weights[i];
        transformed_values[i] = 0.5 * (1.0 + y) / (1.0 - y);
        transformed_weights[i] = wy / ((1.0 - y) * (1.0 - y));
    }

    std::vector<std::size_t> order(num_freq_points);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return transformed_values[a] < transformed_values[b];
    });

    std::vector<double> sorted_values(num_freq_points, 0.0);
    std::vector<double> sorted_weights(num_freq_points, 0.0);
    for (std::size_t i = 0; i < num_freq_points; ++i) {
        sorted_values[i] = transformed_values[order[i]];
        sorted_weights[i] = transformed_weights[order[i]];
    }
    if (!sorted_values.empty() && sorted_values.front() < 1e-12) {
        sorted_values.front() = 1e-12;
    }
    return {sorted_values, sorted_weights};
}

} // namespace gw
