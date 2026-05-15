#include "linalg/linalg_cosma.hpp"

#include "linalg/linalg_scalapack.hpp"
#include "matrix/cosma_distributed_matrix.hpp"
#include "matrix/distributed_matrix.hpp"

#include <stdexcept>
#include <utility>

namespace gw::linalg {
namespace {

[[nodiscard]] std::size_t effective_rows(const MatrixComplex& a, MatrixTranspose trans) noexcept {
    return trans == MatrixTranspose::NoTranspose ? a.rows() : a.cols();
}

[[nodiscard]] std::size_t effective_cols(const MatrixComplex& a, MatrixTranspose trans) noexcept {
    return trans == MatrixTranspose::NoTranspose ? a.cols() : a.rows();
}

[[nodiscard]] MatrixComplex materialize_op(const MatrixComplex& a, MatrixTranspose trans) {
    if (trans == MatrixTranspose::NoTranspose) {
        return a;
    }
    MatrixComplex out(a.cols(), a.rows(), Complex{0.0, 0.0});
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            out(j, i) = a(i, j);
        }
    }
    return out;
}

} // namespace

std::string_view CosmaPxgemmBackend::name() const noexcept {
    return "cosma-pxgemm-wrapper";
}

BackendCapabilities CosmaPxgemmBackend::capabilities() const noexcept {
    BackendCapabilities caps{};
    caps.distributed_mpi = true;
    caps.uses_cosma_pxgemm = true;
#ifdef GW_COSMA_GPU_CAPABLE
    caps.external_provider_may_use_gpu = true;
#endif
    return caps;
}

MatrixComplex CosmaPxgemmBackend::inverse(MatrixComplex a) const {
    // COSMA only replaces distributed GEMM.  Factorization/solve remain
    // ScaLAPACK-owned by design.
    return make_scalapack_backend()->inverse(std::move(a));
}

MatrixComplex CosmaPxgemmBackend::gemm(const MatrixComplex& a,
                                       const MatrixComplex& b,
                                       MatrixTranspose trans_a,
                                       MatrixTranspose trans_b) const {
    if (effective_cols(a, trans_a) != effective_rows(b, trans_b)) {
        throw std::runtime_error("CosmaPxgemmBackend::gemm: incompatible matrix shapes");
    }

    auto grid = matrix::make_blacs_grid();
    auto dist_a = matrix::scatter_replicated_to_distributed(a, grid);
    auto dist_b = matrix::scatter_replicated_to_distributed(b, grid);
    auto dist_c = matrix::cosma_distributed_gemm(dist_a, dist_b, trans_a, trans_b);
    return matrix::gather_distributed_to_replicated(dist_c);
}

std::vector<Complex> CosmaPxgemmBackend::gemv(const MatrixComplex& a,
                                              const std::vector<double>& x,
                                              MatrixTranspose trans_a) const {
    const MatrixComplex op_a = materialize_op(a, trans_a);
    if (x.size() != op_a.cols()) {
        throw std::runtime_error("CosmaPxgemmBackend::gemv: incompatible matrix/vector shapes");
    }

    MatrixComplex xmat(x.size(), 1, Complex{0.0, 0.0});
    for (std::size_t i = 0; i < x.size(); ++i) {
        xmat(i, 0) = Complex{x[i], 0.0};
    }
    const MatrixComplex ymat = gemm(op_a, xmat);
    std::vector<Complex> y(ymat.rows());
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] = ymat(i, 0);
    }
    return y;
}

Complex CosmaPxgemmBackend::quadratic_form(const std::vector<double>& x, const MatrixComplex& a) const {
    if (a.rows() != a.cols() || x.size() != a.rows()) {
        throw std::runtime_error("CosmaPxgemmBackend::quadratic_form: incompatible shapes");
    }
    const std::vector<Complex> ax = gemv(a, x);
    Complex value{0.0, 0.0};
    for (std::size_t i = 0; i < x.size(); ++i) {
        value += x[i] * ax[i];
    }
    return value;
}

std::shared_ptr<const Backend> make_cosma_pxgemm_backend() {
    static const std::shared_ptr<const Backend> backend = std::make_shared<CosmaPxgemmBackend>();
    return backend;
}

} // namespace gw::linalg
