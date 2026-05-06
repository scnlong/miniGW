#pragma once

#include "gw/gw.hpp"

#include <cstddef>

namespace gw {

struct MemoryFootprint {
    long double eri_tensor_bytes{};
    long double mo_energy_bytes{};
    long double vxc_matrix_bytes{};
    long double sigma_x_bytes{};
    long double v_ph_matrix_bytes{};
    long double pq_ph_tensor_bytes{};
    long double sigma_c_bytes{};
    long double frequency_grid_bytes{};
    long double pi0_diag_bytes{};
    long double w_c_per_freq_bytes{};
    long double w_c_workspace_bytes{};

    [[nodiscard]] long double persistent_bytes() const;
    [[nodiscard]] long double peak_bytes() const;
};

MemoryFootprint estimate_memory_footprint(const GwInput& input, std::size_t num_freq_points);
void print_memory_footprint_report(const MemoryFootprint& footprint);

} // namespace gw
