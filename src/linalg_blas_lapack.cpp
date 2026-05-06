#include "gw/linalg_blas_lapack.hpp"

#include <cblas.h>
#include <lapacke.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gw::linalg {
namespace {

[[nodiscard]] std::size_t effective_rows(const MatrixComplex& a, MatrixTranspose trans) noexcept {
    return trans == MatrixTranspose::NoTranspose ? a.rows() : a.cols();
}

[[nodiscard]] std::size_t effective_cols(const MatrixComplex& a, MatrixTranspose trans) noexcept {
    return trans == MatrixTranspose::NoTranspose ? a.cols() : a.rows();
}

[[nodiscard]] CBLAS_TRANSPOSE to_cblas_transpose(MatrixTranspose trans) {
    switch (trans) {
        case MatrixTranspose::NoTranspose:
            return CblasNoTrans;
        case MatrixTranspose::Transpose:
            return CblasTrans;
        case MatrixTranspose::ConjugateTranspose:
            return CblasConjTrans;
    }
    throw std::runtime_error("unknown MatrixTranspose value");
}

[[nodiscard]] int checked_blas_int(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " exceeds the CBLAS int range");
    }
    return static_cast<int>(value);
}

[[nodiscard]] lapack_int checked_lapack_int(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<lapack_int>::max())) {
        throw std::runtime_error(std::string(name) + " exceeds the LAPACK integer range");
    }
    return static_cast<lapack_int>(value);
}

[[nodiscard]] lapack_complex_double* lapack_ptr(std::vector<Complex>& data) noexcept {
    return reinterpret_cast<lapack_complex_double*>(data.data());
}

} // namespace

std::string_view BlasLapackBackend::name() const noexcept {
    return "blas-lapack";
}

BackendCapabilities BlasLapackBackend::capabilities() const noexcept {
    return BackendCapabilities{.thread_safe = true, .uses_internal_threads = true, .distributed_mpi = false, .uses_device_memory = false};
}

MatrixComplex BlasLapackBackend::inverse(MatrixComplex a) const {
    if (a.rows() != a.cols()) {
        throw std::runtime_error("BlasLapackBackend::inverse: matrix must be square");
    }

    const lapack_int n = checked_lapack_int(a.rows(), "matrix dimension");
    const lapack_int lda = n;
    std::vector<lapack_int> ipiv(static_cast<std::size_t>(n));

    lapack_int info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, n, n, lapack_ptr(a.data()), lda, ipiv.data());
    if (info < 0) {
        throw std::runtime_error("LAPACKE_zgetrf: argument " + std::to_string(-info) + " had an illegal value");
    }
    if (info > 0) {
        throw std::runtime_error("LAPACKE_zgetrf: matrix is singular at U(" + std::to_string(info) + "," + std::to_string(info) + ")");
    }

    info = LAPACKE_zgetri(LAPACK_ROW_MAJOR, n, lapack_ptr(a.data()), lda, ipiv.data());
    if (info < 0) {
        throw std::runtime_error("LAPACKE_zgetri: argument " + std::to_string(-info) + " had an illegal value");
    }
    if (info > 0) {
        throw std::runtime_error("LAPACKE_zgetri: matrix is singular at U(" + std::to_string(info) + "," + std::to_string(info) + ")");
    }

    return a;
}

MatrixComplex BlasLapackBackend::gemm(const MatrixComplex& a,
                                      const MatrixComplex& b,
                                      MatrixTranspose trans_a,
                                      MatrixTranspose trans_b) const {
    const std::size_t m = effective_rows(a, trans_a);
    const std::size_t k_a = effective_cols(a, trans_a);
    const std::size_t k_b = effective_rows(b, trans_b);
    const std::size_t n = effective_cols(b, trans_b);
    if (k_a != k_b) {
        throw std::runtime_error("BlasLapackBackend::gemm: incompatible matrix shapes");
    }

    MatrixComplex c(m, n, Complex{0.0, 0.0});
    const Complex alpha{1.0, 0.0};
    const Complex beta{0.0, 0.0};

    cblas_zgemm(CblasRowMajor,
                to_cblas_transpose(trans_a),
                to_cblas_transpose(trans_b),
                checked_blas_int(m, "m"),
                checked_blas_int(n, "n"),
                checked_blas_int(k_a, "k"),
                &alpha,
                a.data().data(),
                checked_blas_int(a.cols(), "lda"),
                b.data().data(),
                checked_blas_int(b.cols(), "ldb"),
                &beta,
                c.data().data(),
                checked_blas_int(c.cols(), "ldc"));

    return c;
}

std::vector<Complex> BlasLapackBackend::gemv(const MatrixComplex& a,
                                             const std::vector<double>& x,
                                             MatrixTranspose trans_a) const {
    const std::size_t m = effective_rows(a, trans_a);
    const std::size_t n = effective_cols(a, trans_a);
    if (x.size() != n) {
        throw std::runtime_error("BlasLapackBackend::gemv: incompatible matrix/vector shapes");
    }

    std::vector<Complex> x_complex(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        x_complex[i] = Complex{x[i], 0.0};
    }

    std::vector<Complex> y(m, Complex{0.0, 0.0});
    const Complex alpha{1.0, 0.0};
    const Complex beta{0.0, 0.0};

    cblas_zgemv(CblasRowMajor,
                to_cblas_transpose(trans_a),
                checked_blas_int(a.rows(), "a.rows"),
                checked_blas_int(a.cols(), "a.cols"),
                &alpha,
                a.data().data(),
                checked_blas_int(a.cols(), "lda"),
                x_complex.data(),
                1,
                &beta,
                y.data(),
                1);

    return y;
}

Complex BlasLapackBackend::quadratic_form(const std::vector<double>& x, const MatrixComplex& a) const {
    if (a.rows() != a.cols() || x.size() != a.rows()) {
        throw std::runtime_error("BlasLapackBackend::quadratic_form: incompatible shapes");
    }

    const std::vector<Complex> ax = gemv(a, x);
    Complex value{0.0, 0.0};
    for (std::size_t i = 0; i < x.size(); ++i) {
        value += x[i] * ax[i];
    }
    return value;
}

std::shared_ptr<const Backend> make_blas_lapack_backend() {
    static const std::shared_ptr<const Backend> backend = std::make_shared<BlasLapackBackend>();
    return backend;
}

} // namespace gw::linalg
