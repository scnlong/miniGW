#pragma once

#include "execution.hpp"
#include "gw/integrals.hpp"
#include "linalg/linalg.hpp"
#include "gw/orbital_space.hpp"
#include "gw/particle_hole.hpp"
#include "gw/sigma.hpp"
#include "types.hpp"

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

    // Number of (p,k) vectors grouped into one panel for the self-energy
    // contraction.  The local path evaluates W_c * P with GEMM, where
    // P contains several pk vectors as columns.  The ScaLAPACK path performs
    // the same contraction over distributed W_c blocks and reduces one panel
    // of scalar results at a time.
    std::size_t contraction_panel_size{32};

    ExecutionPolicy execution{};

	// Fallback backend used when GwSettings is constructed directly in tests.
    // The executable normally overwrites this from the CLI/backend factory.
    std::shared_ptr<const linalg::Backend> linalg_backend{linalg::make_reference_backend()};
};

struct GwTimings {
    double total_wall_seconds{0.0};
    double exchange_seconds{0.0};
    double build_mapping_seconds{0.0};
    double build_inv_v_seconds{0.0};
    double build_pi0_seconds{0.0};        // accumulated thread/rank time in parallel modes
    double invert_epsilon_seconds{0.0};  // accumulated thread/rank time in parallel modes
    double sigma_c_seconds{0.0};         // accumulated thread/rank time in parallel modes
    double sigma_c_wall_seconds{0.0};
    double mpi_reduce_seconds{0.0};
    double pade_seconds{0.0};
};

void add_timings(GwTimings& dst, const GwTimings& src);

struct GwResult {
    MatrixReal sigma_x;
    MatrixComplex sigma_c_im_points;
    std::vector<double> qp_energy;
    std::vector<double> omegas;
    std::vector<double> weights;
    GwTimings timings;
};

GwResult run_g0w0(const GwInput& input, const GwSettings& settings);

} // namespace gw
