#pragma once

#include "types.hpp"

#include <utility>
#include <vector>

namespace gw {

std::pair<std::vector<Complex>, std::vector<Complex>> get_pade_coefficients_continued_fraction(
    const std::vector<Complex>& x,
    const std::vector<Complex>& y,
    std::size_t npar);

Complex evaluate_pade_continued_fraction(
    const std::vector<Complex>& coeffs,
    double z_target,
    const std::vector<Complex>& x_points_for_pade);

} // namespace gw
