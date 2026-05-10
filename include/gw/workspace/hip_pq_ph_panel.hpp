#pragma once

#include "gw/integrals.hpp"
#include "gw/particle_hole.hpp"

#include <cstddef>
#include <memory>
#include <string_view>

namespace gw::workspace {

// Non-owning view of a device-resident complex panel X with column-major layout:
//   shape = n_ph x width
//   X(ph, kk) = (p, k_begin + kk | i a)
// The payload is intentionally exposed as void* so this header stays usable from
// ordinary C++ translation units without including HIP/ROCm headers.
class HipComplexPanelView {
public:
    HipComplexPanelView() = default;

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] const void* device_data() const noexcept { return data_; }
    [[nodiscard]] void* device_data() noexcept { return data_; }

private:
    friend class HipPqPhPanelView;

    HipComplexPanelView(void* data, std::size_t rows, std::size_t cols) noexcept
        : data_(data), rows_(rows), cols_(cols) {}

    void* data_{nullptr};
    std::size_t rows_{0};
    std::size_t cols_{0};
};

enum class HipPqPhPanelStorageMode {
    Auto,
    FullEriResident,
    StreamingPanel
};

[[nodiscard]] std::string_view to_string(HipPqPhPanelStorageMode mode) noexcept;

// HIP/ROCm-resident/streaming panel source for P_{p k, i a} = (p k | i a).
//
// Two modes are supported:
//
//   FullEriResident:
//     Upload the full replicated ERI tensor to one GPU and assemble each panel
//     with a HIP/ROCm kernel.  This avoids per-panel H2D traffic but can exhaust
//     GPU memory for realistic nmo.
//
//   StreamingPanel:
//     Keep ERI on the host, assemble only the requested n_ph x panel_width slice
//     into a pinned host staging buffer, and copy that panel to the GPU.  This
//     is not as fast as full device residency, but it avoids a full nmo^4 ERI
//     copy on the GPU and is the safer dense single-GPU path.
//
//   Auto:
//     Choose FullEriResident only if the ERI copy fits a conservative fraction
//     of currently available device memory; otherwise use StreamingPanel.
//
// This class still does not implement distributed/tiled ERI ownership.  It is a
// final dense single-GPU bridge before a genuinely integral-driven/tiled GW
// implementation.
class HipPqPhPanelView {
public:
    HipPqPhPanelView(const MolecularIntegrals& integrals,
                        const ParticleHoleBasis& ph_basis,
                        std::size_t max_panel_width,
                        HipPqPhPanelStorageMode mode = HipPqPhPanelStorageMode::Auto);
    ~HipPqPhPanelView();

    HipPqPhPanelView(const HipPqPhPanelView&) = delete;
    HipPqPhPanelView& operator=(const HipPqPhPanelView&) = delete;

    HipPqPhPanelView(HipPqPhPanelView&&) noexcept;
    HipPqPhPanelView& operator=(HipPqPhPanelView&&) noexcept;

    [[nodiscard]] std::size_t nmo() const noexcept;
    [[nodiscard]] std::size_t nph() const noexcept;
    [[nodiscard]] std::size_t max_panel_width() const noexcept;
    [[nodiscard]] std::size_t last_panel_width() const noexcept;
    [[nodiscard]] std::size_t estimated_device_bytes() const noexcept;
    [[nodiscard]] std::size_t estimated_host_pinned_bytes() const noexcept;
    [[nodiscard]] HipPqPhPanelStorageMode storage_mode() const noexcept;
    [[nodiscard]] std::string_view storage_mode_name() const noexcept;

    // Fill and return a non-owning view of the internal device panel buffer.
    // The view remains valid until the next call to fill_panel() or destruction.
    [[nodiscard]] HipComplexPanelView fill_panel(std::size_t p_index,
                                                    std::size_t k_begin,
                                                    std::size_t width);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gw::workspace
