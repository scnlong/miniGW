#pragma once

#include "gw/integrals.hpp"
#include "gw/orbital_space.hpp"
#include "gw/particle_hole.hpp"
#include "linalg/linalg.hpp"
#include "types.hpp"

#include <vector>

namespace gw {

void set_sigma_openmp_kernel_loops(bool enabled) noexcept;

MatrixReal calculate_exchange(const OrbitalSpace& orbitals, const MolecularIntegrals& integrals);
std::vector<Complex> calculate_pi0_ph_diag(Complex omega, const ParticleHoleBasis& ph_basis, double eta);
MatrixReal calculate_v_ph_matrix(const MolecularIntegrals& integrals, const ParticleHoleBasis& ph_basis);
MatrixComplex calculate_w_0_c_matrix(const MatrixReal& v_ph,
                                     const MatrixComplex& inv_v,
                                     const std::vector<Complex>& pi0_diag,
                                     const linalg::Backend& backend = linalg::reference_backend());
Tensor3Real calculate_pq_ph_matrix(const MolecularIntegrals& integrals, const ParticleHoleBasis& ph_basis);

} // namespace gw
