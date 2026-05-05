#pragma once

#include <utility>
#include <vector>

namespace gw {

std::pair<std::vector<double>, std::vector<double>> generate_linear_grid(std::size_t num_freq_points, double max_freq);
std::pair<std::vector<double>, std::vector<double>> generate_transformed_legendre_grid(std::size_t num_freq_points);

} // namespace gw
