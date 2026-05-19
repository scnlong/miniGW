#include "workspace/cosma_distributed_screening_workspace.hpp"

#include "matrix/cosma_distributed_matrix.hpp"

#include <stdexcept>

namespace gw::workspace {

CosmaDistributedScreeningWorkspace::CosmaDistributedScreeningWorkspace(const MolecularIntegrals& integrals,
                                                                       const ParticleHoleBasis& ph_basis,
                                                                       int block_size,
                                                                       std::size_t ranks_per_group)
    : grid_(matrix::make_blacs_grid(static_cast<int>(ranks_per_group))),
      v_ph_(grid_, ph_basis.size(), ph_basis.size(), block_size, block_size),
      w_c_(grid_, ph_basis.size(), ph_basis.size(), block_size, block_size),
      block_size_(block_size) {
    v_ph_.for_each_owned_global([&](std::size_t row, std::size_t col, Complex& value) {
        const ParticleHolePair& jb = ph_basis[row];
        const ParticleHolePair& ia = ph_basis[col];
        value = Complex{integrals.eri(jb.i_occ, jb.a_mo, ia.i_occ, ia.a_mo), 0.0};
    });
}

void CosmaDistributedScreeningWorkspace::compute_w_c(const std::vector<Complex>& pi0_diag) {
    if (pi0_diag.size() != v_ph_.global_rows()) {
        throw std::runtime_error("CosmaDistributedScreeningWorkspace::compute_w_c: inconsistent pi0 dimension");
    }

    // Use the same stable left-dielectric construction as the host and
    // ScaLAPACK workspaces:
    //
    //     W_c = (I - D V_ph)^(-1) D,   D = diag(pi0_diag).
    //
    // This avoids explicitly constructing V_ph^(-1).  COSMA is still used for
    // the W_c * X panel contractions below; W_c construction itself is a
    // distributed solve because it forms the inverse of the dielectric matrix.
    matrix::DistributedMatrixComplex epsilon_left(grid_, v_ph_.global_rows(), v_ph_.global_cols(), block_size_, block_size_);
    epsilon_left.for_each_owned_global([&](std::size_t row, std::size_t col, Complex& value) {
        value = -pi0_diag[row] * v_ph_.owned_global_at(row, col);
        if (row == col) {
            value += Complex{1.0, 0.0};
        }
    });

    w_c_ = matrix::distributed_inverse_by_solve(epsilon_left);
    w_c_.for_each_owned_global([&](std::size_t row, std::size_t col, Complex& value) {
        (void)row;
        value *= pi0_diag[col];
    });
}

Complex CosmaDistributedScreeningWorkspace::quadratic_form(const std::vector<double>& x) const {
    if (x.size() != w_c_.global_rows() || w_c_.global_rows() != w_c_.global_cols()) {
        throw std::runtime_error("CosmaDistributedScreeningWorkspace::quadratic_form: inconsistent vector/matrix dimensions");
    }

    Complex local_sum{0.0, 0.0};
    w_c_.for_each_owned_global([&](std::size_t row, std::size_t col, const Complex& value) {
        local_sum += x[row] * value * x[col];
    });
    return matrix::mpi_allreduce_sum_complex(local_sum, *grid_);
}

std::vector<Complex> CosmaDistributedScreeningWorkspace::quadratic_forms_panel(const MatrixReal& x_panel) const {
    if (x_panel.rows() != w_c_.global_rows() || w_c_.global_rows() != w_c_.global_cols()) {
        throw std::runtime_error("CosmaDistributedScreeningWorkspace::quadratic_forms_panel: inconsistent vector/matrix dimensions");
    }

    matrix::DistributedMatrixComplex x_dist =
        matrix::scatter_real_panel_to_distributed(x_panel, grid_, block_size_);
    matrix::DistributedMatrixComplex y_dist =
        matrix::cosma_distributed_gemm(w_c_,
                                       x_dist,
                                       linalg::MatrixTranspose::NoTranspose,
                                       linalg::MatrixTranspose::NoTranspose);
    return matrix::distributed_column_dot_same_layout(x_dist, y_dist);
}

} // namespace gw::workspace
