#pragma once

#include "linalg/linalg.hpp"
#include "matrix/ownership.hpp"
#include "types.hpp"

#include <vector>

namespace gw::workspace {

void set_host_screening_openmp_kernel_loops(bool enabled) noexcept;

class HostScreeningWorkspace {
public:
    HostScreeningWorkspace(MatrixReal v_ph, MatrixComplex inv_v, const linalg::Backend& backend);

    [[nodiscard]] const MatrixReal& v_ph() const noexcept { return v_ph_; }
    [[nodiscard]] const MatrixComplex& inv_v() const noexcept { return inv_v_; }
    [[nodiscard]] const linalg::Backend& backend() const noexcept { return backend_; }
    [[nodiscard]] matrix::DataOwnership ownership() const noexcept { return ownership_; }

    [[nodiscard]] MatrixComplex compute_w_c(const std::vector<Complex>& pi0_diag) const;

    // Evaluate x_j^T W_c x_j for a panel of vectors.  x_panel has shape
    // n_ph x nvec with each column one pk vector.  The local implementation
    // computes Y = W_c * X with the selected backend GEMM and then contracts
    // column-wise.
    [[nodiscard]] std::vector<Complex> quadratic_forms_panel(const MatrixComplex& w_c,
                                                             const MatrixReal& x_panel) const;

private:
    MatrixReal v_ph_;
    MatrixComplex inv_v_;
    const linalg::Backend& backend_;
    matrix::DataOwnership ownership_{matrix::replicated_host_ownership()};
};

} // namespace gw::workspace
