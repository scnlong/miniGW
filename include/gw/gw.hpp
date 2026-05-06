#pragma once

#include "gw/integrals.hpp"
#include "gw/linalg.hpp"
#include "gw/orbital_space.hpp"
#include "gw/particle_hole.hpp"
#include "gw/types.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace gw {

struct GwInput {
    Tensor4Real eri_mo;
    std::vector<double> mo_energy;
    MatrixReal vxc_mo;
    std::size_t nocc{0};
    double fermi_energy{0.0};
};

struct GwSettings {
    std::size_t num_freq_points_total{200};
    std::size_t num_pade_params{16};
    std::optional<std::size_t> selected_state_0based{4}; // Default cal_states = 5.
    double eta{0.0};

    // Keep the numerical backend explicit. The default backend is a
    // serial reference implementation; optimized OpenBLAS/LAPACK,
    // ScaLAPACK/COSMA, cuBLAS/cuSolver, or HIP backends can implement
    // gw::linalg::Backend without changing the GW equations below.
    std::shared_ptr<const linalg::Backend> linalg_backend{linalg::make_reference_backend()};
};

struct GwTimings {
    double exchange_seconds{0.0};
    double build_mapping_seconds{0.0};
    double build_pi0_seconds{0.0};
    double invert_epsilon_seconds{0.0};
    double sigma_c_seconds{0.0};
    double pade_seconds{0.0};
};

struct GwResult {
    MatrixReal sigma_x;
    MatrixComplex sigma_c_im_points;
    std::vector<double> qp_energy;
    std::vector<double> omegas;
    std::vector<double> weights;
    GwTimings timings;
};

MatrixReal calculate_exchange(const OrbitalSpace& orbitals, const MolecularIntegrals& integrals);
std::vector<Complex> calculate_pi0_ph_diag(Complex omega, const ParticleHoleBasis& ph_basis, double eta);
MatrixReal calculate_v_ph_matrix(const MolecularIntegrals& integrals, const ParticleHoleBasis& ph_basis);
MatrixComplex calculate_w_0_c_matrix(Complex omega, const MatrixReal& v_ph, const std::vector<Complex>& pi0_diag, const linalg::Backend& backend = linalg::reference_backend());
Tensor3Real calculate_pq_ph_matrix(const MolecularIntegrals& integrals, const ParticleHoleBasis& ph_basis);
GwResult run_g0w0(const GwInput& input, const GwSettings& settings);

} // namespace gw
