#include "gw/workspace/distributed_screening_workspace.hpp"

#include <stdexcept>

namespace gw::workspace {

DistributedScreeningWorkspace::DistributedScreeningWorkspace(const MolecularIntegrals& integrals,
                                                             const ParticleHoleBasis& ph_basis,
                                                             int block_size,
                                                             matrix::DistributedGemmProvider gemm_provider,
                                                             std::size_t ranks_per_group)
    : grid_(matrix::make_blacs_grid(static_cast<int>(ranks_per_group))),
      v_ph_(grid_, ph_basis.size(), ph_basis.size(), block_size, block_size),
      inv_v_(grid_, ph_basis.size(), ph_basis.size(), block_size, block_size),
      w_c_(grid_, ph_basis.size(), ph_basis.size(), block_size, block_size),
      block_size_(block_size),
      gemm_provider_(gemm_provider) {
    v_ph_.for_each_owned_global([&](std::size_t row, std::size_t col, Complex& value) {
        const ParticleHolePair& jb = ph_basis[row];
        const ParticleHolePair& ia = ph_basis[col];
        value = Complex{integrals.eri(jb.i_occ, jb.a_mo, ia.i_occ, ia.a_mo), 0.0};
    });

    inv_v_ = matrix::distributed_inverse_by_solve(v_ph_);
}

void DistributedScreeningWorkspace::compute_w_c(const std::vector<Complex>& pi0_diag) {
    if (pi0_diag.size() != v_ph_.global_rows()) {
        throw std::runtime_error("DistributedScreeningWorkspace::compute_w_c: inconsistent pi0 dimension");
    }

    matrix::DistributedMatrixComplex epsilon(grid_, v_ph_.global_rows(), v_ph_.global_cols(), block_size_, block_size_);
    epsilon.for_each_owned_global([&](std::size_t row, std::size_t col, Complex& value) {
        value = -v_ph_.owned_global_at(row, col) * pi0_diag[col];
        if (row == col) {
            value += Complex{1.0, 0.0};
        }
    });

    matrix::DistributedMatrixComplex inv_eps = matrix::distributed_inverse_by_solve(epsilon);
    inv_eps.for_each_owned_global([](std::size_t row, std::size_t col, Complex& value) {
        if (row == col) {
            value -= Complex{1.0, 0.0};
        }
    });

    w_c_ = matrix::distributed_gemm(inv_eps,
                                    inv_v_,
                                    linalg::MatrixTranspose::Transpose,
                                    linalg::MatrixTranspose::NoTranspose,
                                    gemm_provider_);
}

Complex DistributedScreeningWorkspace::quadratic_form(const std::vector<double>& x) const {
    if (x.size() != w_c_.global_rows() || w_c_.global_rows() != w_c_.global_cols()) {
        throw std::runtime_error("DistributedScreeningWorkspace::quadratic_form: inconsistent vector/matrix dimensions");
    }

    Complex local_sum{0.0, 0.0};
    w_c_.for_each_owned_global([&](std::size_t row, std::size_t col, const Complex& value) {
        local_sum += x[row] * value * x[col];
    });
    return matrix::mpi_allreduce_sum_complex(local_sum, *grid_);
}

std::vector<Complex> DistributedScreeningWorkspace::quadratic_forms_panel(const MatrixReal& x_panel) const {
    if (x_panel.rows() != w_c_.global_rows() || w_c_.global_rows() != w_c_.global_cols()) {
        throw std::runtime_error("DistributedScreeningWorkspace::quadratic_forms_panel: inconsistent vector/matrix dimensions");
    }

    // Distributed panel contraction:
    //   X = [x_1, ..., x_m],  X is n_ph x m
    //   Y = W_c X              PZGEMM on block-cyclic matrices
    //   q_j = x_j^T y_j        local column dot followed by one vector Allreduce
    // This replaces the previous local O(n_ph^2 * m) loop over the replicated
    // panel with a true distributed GEMM.  The input panel is still assembled
    // from the replicated pq_ph tensor in the current code; the ownership of the
    // dense W_c application is now distributed.
    matrix::DistributedMatrixComplex x_dist =
        matrix::scatter_real_panel_to_distributed(x_panel, grid_, block_size_);
    matrix::DistributedMatrixComplex y_dist =
        matrix::distributed_gemm(w_c_,
                                 x_dist,
                                 linalg::MatrixTranspose::NoTranspose,
                                 linalg::MatrixTranspose::NoTranspose,
                                 gemm_provider_);
    return matrix::distributed_column_dot_same_layout(x_dist, y_dist);
}

} // namespace gw::workspace
