#pragma once

#include "gw/gw.hpp"
#include "gw/integrals.hpp"
#include "gw/orbital_space.hpp"
#include "gw/particle_hole.hpp"
#include "linalg/linalg.hpp"
#include "types.hpp"

#include <cstddef>
#include <vector>

namespace gw {

void compute_sigma_c_with_backend(const OrbitalSpace& orbitals,
                                  const MolecularIntegrals& integrals,
                                  const ParticleHoleBasis& ph_basis,
                                  const std::vector<Complex>& omega_im,
                                  const std::vector<double>& weights,
                                  const std::vector<std::size_t>& states,
                                  const GwSettings& settings,
                                  const linalg::Backend& linalg_backend,
                                  MatrixComplex& sigma_c_im_points,
                                  GwTimings& timings);

} // namespace gw
