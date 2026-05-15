#pragma once

#include "gw/gw.hpp"
#include "gw/integrals.hpp"
#include "gw/orbital_space.hpp"
#include "types.hpp"

#include <cstddef>
#include <vector>

namespace gw {

std::vector<double> solve_qp_energies(const OrbitalSpace& orbitals,
                                      const MolecularIntegrals& integrals,
                                      const MatrixReal& sigma_x,
                                      const MatrixComplex& sigma_c_im_points,
                                      const std::vector<Complex>& omega_im,
                                      const std::vector<std::size_t>& states,
                                      const GwSettings& settings);

} // namespace gw
