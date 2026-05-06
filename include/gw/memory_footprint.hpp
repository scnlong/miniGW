#pragma once

#include "gw/gw.hpp"

#include <cstddef>

namespace gw {

struct MemoryFootprint {
    std::size_t nmo{};
    std::size_t nocc{};
    std::size_t nvirt{};
    std::size_t n_ph{};
    std::size_t num_freq_points{};

    bool mpi_enabled{};
    std::size_t mpi_size{1};
    FrequencyParallelMode frequency_parallel_mode{FrequencyParallelMode::Serial};
    std::size_t frequency_workspace_replicas{1};

    // Input arrays loaded into GwInput.
    long double input_eri_bytes{};
    long double input_vxc_bytes{};
    long double input_mo_energy_bytes{};

    // Extra resident copies made by the current MolecularIntegrals ownership model.
    // If MolecularIntegrals is later changed to hold const references/views, these
    // two terms should be set to zero.
    long double integrals_eri_copy_bytes{};
    long double integrals_vxc_copy_bytes{};

    // Main resident arrays allocated by the current G0W0 workflow.
    long double sigma_x_bytes{};
    long double v_ph_matrix_bytes{};
    long double pq_ph_tensor_bytes{};
    long double sigma_c_bytes{};
    long double qp_energy_bytes{};
    long double frequency_grid_bytes{};

    // Per-frequency small work arrays.
    long double inv_v_bytes{};
    long double pi0_diag_bytes{};
    long double pk_vec_bytes{};
    long double current_sigma_bytes{};
    long double quadratic_form_workspace_bytes{};

    // Per-frequency dense W_c construction peak. This is deliberately a peak term,
    // not a persistent term, because W_c is produced and consumed inside one
    // frequency-point iteration.
    long double w_c_matrix_bytes{};
    long double w_c_construction_peak_bytes{};

    [[nodiscard]] long double input_bytes() const;
    [[nodiscard]] long double runtime_copy_bytes() const;
    [[nodiscard]] long double persistent_runtime_bytes_per_rank() const;
    [[nodiscard]] long double per_frequency_workspace_bytes_one_replica() const;
    [[nodiscard]] long double replicated_workspace_bytes_per_rank() const;
    [[nodiscard]] long double peak_bytes_per_rank() const;
    [[nodiscard]] long double aggregate_peak_bytes_all_ranks() const;
};

[[nodiscard]] MemoryFootprint estimate_memory_footprint(const GwInput& input,
                                                        const GwSettings& settings);
[[nodiscard]] MemoryFootprint estimate_memory_footprint(const GwInput& input,
                                                        std::size_t num_freq_points);
void print_memory_footprint_report(const MemoryFootprint& footprint);

} // namespace gw
