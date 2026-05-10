#pragma once

#include "gw/matrix/ownership.hpp"
#include "gw/types.hpp"
#include "gw/workspace/hip_pq_ph_panel.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace gw::workspace {

// HIP/ROCm device-resident screening workspace.
//
// This class is intentionally independent of the generic linalg::Backend host
// interface.  The generic cuBLAS backend accepts and returns replicated host
// matrices, which is useful for correctness testing but causes repeated
// hipMalloc/hipMemcpy/hipFree traffic.  HipScreeningWorkspace keeps the
// screening matrices and solver workspaces on the GPU across frequencies:
//
//   V_ph(device), inv(V_ph)(device), epsilon(device), W_c(device)
//
// Per frequency it uploads only pi0_diag, rebuilds epsilon on the device,
// factorizes/solves there, and keeps W_c resident for panel contractions.
class HipScreeningWorkspace {
public:
    explicit HipScreeningWorkspace(const MatrixReal& v_ph);
    ~HipScreeningWorkspace();

    HipScreeningWorkspace(const HipScreeningWorkspace&) = delete;
    HipScreeningWorkspace& operator=(const HipScreeningWorkspace&) = delete;

    HipScreeningWorkspace(HipScreeningWorkspace&&) noexcept;
    HipScreeningWorkspace& operator=(HipScreeningWorkspace&&) noexcept;

    [[nodiscard]] std::size_t nph() const noexcept;
    [[nodiscard]] std::size_t last_panel_width() const noexcept;
    [[nodiscard]] std::size_t estimated_device_bytes() const noexcept;
    [[nodiscard]] matrix::DataOwnership ownership() const noexcept;

    void compute_w_c(const std::vector<Complex>& pi0_diag);

    // Evaluate x_j^T W_c x_j for a panel X.  x_panel is row-major host data
    // with shape n_ph x nvec.  The panel is copied to the GPU, Y = W_c X is
    // computed with cuBLAS, and one small vector of nvec quadratic forms is
    // copied back to the host.
    [[nodiscard]] std::vector<Complex> quadratic_forms_panel(const MatrixReal& x_panel);

    // Same contraction as above, but X is already assembled in device memory
    // by HipPqPhPanelView.  This is the HIP/ROCm fast path that avoids the
    // host pk_panel allocation and H2D copy per contraction panel.
    [[nodiscard]] std::vector<Complex> quadratic_forms_panel(const HipComplexPanelView& x_panel);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gw::workspace
