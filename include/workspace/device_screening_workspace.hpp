#pragma once

#include "matrix/ownership.hpp"
#include "types.hpp"
#include "workspace/device_pq_ph_panel.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace gw::workspace {

// CUDA device-resident screening workspace.
//
// This class is intentionally independent of the generic linalg::Backend host
// interface.  The generic cuBLAS backend accepts and returns replicated host
// matrices, which is useful for correctness testing but causes repeated
// cudaMalloc/cudaMemcpy/cudaFree traffic.  DeviceScreeningWorkspace keeps the
// screening matrices and solver workspaces on the GPU across frequencies:
//
//   V_ph(device), epsilon_left(device), W_c(device)
//
// Per frequency it uploads only pi0_diag, rebuilds
// epsilon_left = I - diag(Pi0) V_ph on the device, solves for
// W_c = epsilon_left^{-1} diag(Pi0), and keeps W_c resident for panel
// contractions.  It does not explicitly construct inv(V_ph).
class DeviceScreeningWorkspace {
public:
    explicit DeviceScreeningWorkspace(const MatrixReal& v_ph);
    ~DeviceScreeningWorkspace();

    DeviceScreeningWorkspace(const DeviceScreeningWorkspace&) = delete;
    DeviceScreeningWorkspace& operator=(const DeviceScreeningWorkspace&) = delete;

    DeviceScreeningWorkspace(DeviceScreeningWorkspace&&) noexcept;
    DeviceScreeningWorkspace& operator=(DeviceScreeningWorkspace&&) noexcept;

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
    // by DevicePqPhPanelView.  This is the CUDA fast path that avoids the
    // host pk_panel allocation and H2D copy per contraction panel.
    [[nodiscard]] std::vector<Complex> quadratic_forms_panel(const DeviceComplexPanelView& x_panel);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gw::workspace
