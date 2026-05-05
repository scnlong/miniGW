#pragma once

#include "gw/types.hpp"

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
};

struct GwResult {
    MatrixReal sigma_x;
    MatrixComplex sigma_c_im_points;
    std::vector<double> qp_energy;
    std::vector<double> omegas;
    std::vector<double> weights;
};

MatrixReal calculate_exchange(const Tensor4Real& eri_mo, std::size_t nocc);
std::vector<Complex> calculate_pi0_ph_diag(Complex omega, const std::vector<double>& mo_energy, std::size_t nocc, std::size_t nvirt, double eta);
MatrixReal calculate_v_ph_matrix(const Tensor4Real& eri_mo, std::size_t nocc, std::size_t nvirt);
MatrixComplex calculate_w_0_c_matrix(Complex omega, const MatrixReal& v_ph, const std::vector<Complex>& pi0_diag);
Tensor3Real calculate_pq_ph_matrix(const Tensor4Real& eri_mo, std::size_t nocc, std::size_t nvirt);
GwResult run_g0w0(const GwInput& input, const GwSettings& settings);

} // namespace gw
