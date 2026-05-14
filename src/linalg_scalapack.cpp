#include "gw/linalg_scalapack.hpp"

#include "gw/matrix/distributed_matrix.hpp"

#include <complex>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace gw::linalg {
namespace {

[[nodiscard]] std::size_t effective_rows(const MatrixComplex& a, MatrixTranspose trans) noexcept {
    return trans == MatrixTranspose::NoTranspose ? a.rows() : a.cols();
}

[[nodiscard]] std::size_t effective_cols(const MatrixComplex& a, MatrixTranspose trans) noexcept {
    return trans == MatrixTranspose::NoTranspose ? a.cols() : a.rows();
}

[[nodiscard]] Complex access(const MatrixComplex& a, std::size_t i, std::size_t j, MatrixTranspose trans) {
    switch (trans) {
        case MatrixTranspose::NoTranspose:
            return a(i, j);
        case MatrixTranspose::Transpose:
            return a(j, i);
        case MatrixTranspose::ConjugateTranspose:
            return std::conj(a(j, i));
    }
    throw std::runtime_error("unknown MatrixTranspose value");
}

[[nodiscard]] MatrixComplex materialize_op(const MatrixComplex& a, MatrixTranspose trans) {
    MatrixComplex out(effective_rows(a, trans), effective_cols(a, trans), Complex{0.0, 0.0});
    for (std::size_t i = 0; i < out.rows(); ++i) {
        for (std::size_t j = 0; j < out.cols(); ++j) {
            out(i, j) = access(a, i, j, trans);
        }
    }
    return out;
}

} // namespace

std::string_view ScalapackBackend::name() const noexcept {
    return "scalapack-replicated-wrapper";
}

BackendCapabilities ScalapackBackend::capabilities() const noexcept {
    return BackendCapabilities{.family = BackendFamily::DistributedHost,
                               .thread_safe = true,
                               .uses_internal_threads = false,
                               .distributed_mpi = true,
                               .uses_device_memory = false};
}

MatrixComplex ScalapackBackend::inverse(MatrixComplex a) const {
    if (a.rows() != a.cols()) {
        throw std::runtime_error("ScalapackBackend::inverse: matrix must be square");
    }
    auto grid = matrix::make_blacs_grid();
    auto dist_a = matrix::scatter_replicated_to_distributed(a, grid);
    auto dist_inv = matrix::distributed_inverse_by_solve(dist_a);
    return matrix::gather_distributed_to_replicated(dist_inv);
}

MatrixComplex ScalapackBackend::gemm(const MatrixComplex& a,
                                     const MatrixComplex& b,
                                     MatrixTranspose trans_a,
                                     MatrixTranspose trans_b) const {
    // This replicated-wrapper path is kept for integration tests and legacy
    // calls.  The production distributed screening path uses
    // DistributedMatrixComplex directly and avoids gather/scatter in the GW loop.
    const MatrixComplex op_a = materialize_op(a, trans_a);
    const MatrixComplex op_b = materialize_op(b, trans_b);
    if (op_a.cols() != op_b.rows()) {
        throw std::runtime_error("ScalapackBackend::gemm: incompatible matrix shapes");
    }

    auto grid = matrix::make_blacs_grid();
    auto dist_a = matrix::scatter_replicated_to_distributed(op_a, grid);
    auto dist_b = matrix::scatter_replicated_to_distributed(op_b, grid);
    auto dist_c = matrix::distributed_gemm(dist_a, dist_b);
    return matrix::gather_distributed_to_replicated(dist_c);
}

std::vector<Complex> ScalapackBackend::gemv(const MatrixComplex& a,
                                            const std::vector<double>& x,
                                            MatrixTranspose trans_a) const {
    const MatrixComplex op_a = materialize_op(a, trans_a);
    if (x.size() != op_a.cols()) {
        throw std::runtime_error("ScalapackBackend::gemv: incompatible matrix/vector shapes");
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

Complex ScalapackBackend::quadratic_form(const std::vector<double>& x, const MatrixComplex& a) const {
    if (a.rows() != a.cols() || x.size() != a.rows()) {
        throw std::runtime_error("ScalapackBackend::quadratic_form: incompatible shapes");
    }
    const std::vector<Complex> ax = gemv(a, x);
    Complex value{0.0, 0.0};
    for (std::size_t i = 0; i < x.size(); ++i) {
        value += x[i] * ax[i];
    }
    return value;
}

std::shared_ptr<const Backend> make_scalapack_backend() {
    static const std::shared_ptr<const Backend> backend = std::make_shared<ScalapackBackend>();
    return backend;
}

} // namespace gw::linalg
