#pragma once

#include "gw/integrals.hpp"
#include "gw/matrix/distributed_matrix.hpp"
#include "gw/particle_hole.hpp"
#include "gw/types.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace gw::workspace {

// True distributed-memory screening workspace for ScaLAPACK-style execution.
// V_ph, epsilon, inv(V_ph), and W_c are owned as 2-D BLACS block-cyclic matrices;
// W_c is not gathered during the self-energy contraction.  The current project
// still keeps ERI and pq_ph replicated; this class removes the main n_ph^2
// screening matrices from per-rank replicated ownership.
class DistributedScreeningWorkspace {
public:
    DistributedScreeningWorkspace(const MolecularIntegrals& integrals,
                                  const ParticleHoleBasis& ph_basis,
                                  int block_size = 64,
                                  std::size_t ranks_per_group = 0,
                                  matrix::DistributedGemmProvider gemm_provider = matrix::DistributedGemmProvider::Scalapack);

    [[nodiscard]] std::size_t size() const noexcept { return v_ph_.global_rows(); }
    [[nodiscard]] const matrix::DistributedMatrixComplex& v_ph() const noexcept { return v_ph_; }
    [[nodiscard]] const matrix::DistributedMatrixComplex& inv_v() const noexcept { return inv_v_; }
    [[nodiscard]] const matrix::DistributedMatrixComplex& w_c() const noexcept { return w_c_; }
    [[nodiscard]] const std::shared_ptr<const matrix::BlacsGrid>& grid() const noexcept { return grid_; }

    void compute_w_c(const std::vector<Complex>& pi0_diag);
    [[nodiscard]] Complex quadratic_form(const std::vector<double>& x) const;

    // Evaluate x_j^T W_c x_j for a panel of vectors. x_panel has shape
    // n_ph x nvec. The panel is distributed block-cyclic, then Y = W_c * X is
    // computed with ScaLAPACK PZGEMM, and only the nvec scalar contractions are
    // reduced. This is the scalable contraction path for distributed screening.
    [[nodiscard]] std::vector<Complex> quadratic_forms_panel(const MatrixReal& x_panel) const;

private:
    std::shared_ptr<const matrix::BlacsGrid> grid_{};
    matrix::DistributedMatrixComplex v_ph_{};
    matrix::DistributedMatrixComplex inv_v_{};
    matrix::DistributedMatrixComplex w_c_{};
    int block_size_{64};
    matrix::DistributedGemmProvider gemm_provider_{matrix::DistributedGemmProvider::Scalapack};
};

} // namespace gw::workspace
