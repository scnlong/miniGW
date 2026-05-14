#pragma once

#include "gw/integrals.hpp"
#include "gw/matrix/distributed_matrix.hpp"
#include "gw/particle_hole.hpp"
#include "gw/types.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace gw::workspace {

// COSMA variant of the distributed screening workspace.
//
// The storage layout and ScaLAPACK factorization/solve path are identical to
// DistributedScreeningWorkspace, but all dense distributed GEMM applications in
// W_c construction and panel contractions call matrix::cosma_distributed_gemm,
// which in turn calls cosma_pzgemm_.
class CosmaDistributedScreeningWorkspace {
public:
    CosmaDistributedScreeningWorkspace(const MolecularIntegrals& integrals,
                                       const ParticleHoleBasis& ph_basis,
                                       int block_size = 64,
                                       std::size_t ranks_per_group = 0);

    [[nodiscard]] std::size_t size() const noexcept { return v_ph_.global_rows(); }
    [[nodiscard]] const matrix::DistributedMatrixComplex& v_ph() const noexcept { return v_ph_; }
    [[nodiscard]] const matrix::DistributedMatrixComplex& inv_v() const noexcept { return inv_v_; }
    [[nodiscard]] const matrix::DistributedMatrixComplex& w_c() const noexcept { return w_c_; }
    [[nodiscard]] const std::shared_ptr<const matrix::BlacsGrid>& grid() const noexcept { return grid_; }

    void compute_w_c(const std::vector<Complex>& pi0_diag);
    [[nodiscard]] Complex quadratic_form(const std::vector<double>& x) const;
    [[nodiscard]] std::vector<Complex> quadratic_forms_panel(const MatrixReal& x_panel) const;

private:
    std::shared_ptr<const matrix::BlacsGrid> grid_{};
    matrix::DistributedMatrixComplex v_ph_{};
    matrix::DistributedMatrixComplex inv_v_{};
    matrix::DistributedMatrixComplex w_c_{};
    int block_size_{64};
};

} // namespace gw::workspace
