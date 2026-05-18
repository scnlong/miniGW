#include "linalg/linalg_blas_lapack.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
void zgemm_(const char* transa,
            const char* transb,
            const int* m,
            const int* n,
            const int* k,
            const gw::Complex* alpha,
            const gw::Complex* a,
            const int* lda,
            const gw::Complex* b,
            const int* ldb,
            const gw::Complex* beta,
            gw::Complex* c,
            const int* ldc);
void zgemv_(const char* trans,
            const int* m,
            const int* n,
            const gw::Complex* alpha,
            const gw::Complex* a,
            const int* lda,
            const gw::Complex* x,
            const int* incx,
            const gw::Complex* beta,
            gw::Complex* y,
            const int* incy);
void zgetrf_(const int* m, const int* n, gw::Complex* a, const int* lda, int* ipiv, int* info);
void zgetrs_(const char* trans,
             const int* n,
             const int* nrhs,
             const gw::Complex* a,
             const int* lda,
             const int* ipiv,
             gw::Complex* b,
             const int* ldb,
             int* info);
}

namespace gw::linalg {
namespace {

[[nodiscard]] std::size_t effective_rows(const MatrixComplex& a, MatrixTranspose trans) noexcept {
    return trans == MatrixTranspose::NoTranspose ? a.rows() : a.cols();
}

[[nodiscard]] std::size_t effective_cols(const MatrixComplex& a, MatrixTranspose trans) noexcept {
    return trans == MatrixTranspose::NoTranspose ? a.cols() : a.rows();
}

[[nodiscard]] char to_fortran_transpose(MatrixTranspose trans) {
    switch (trans) {
        case MatrixTranspose::NoTranspose:
            return 'N';
        case MatrixTranspose::Transpose:
            return 'T';
        case MatrixTranspose::ConjugateTranspose:
            return 'C';
    }
    throw std::runtime_error("unknown MatrixTranspose value");
}

[[nodiscard]] int checked_blas_int(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " exceeds the BLAS integer range");
    }
    return static_cast<int>(value);
}

[[nodiscard]] int checked_lapack_int(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " exceeds the LAPACK integer range");
    }
    return static_cast<int>(value);
}

void validate_lu_pivots(const std::vector<int>& ipiv, int n) {
    for (std::size_t i = 0; i < ipiv.size(); ++i) {
        if (ipiv[i] < 1 || ipiv[i] > n) {
            throw std::runtime_error(
                "zgetrf_: pivot array contains an invalid entry at position " +
                std::to_string(i) + ": ipiv=" + std::to_string(ipiv[i]) +
                ", expected a 1-based index in [1," + std::to_string(n) +
                "]. This usually indicates a broken BLAS/LAPACK integer ABI "
                "or mixed runtime-library linkage.");
        }
    }
}

} // namespace

std::string_view BlasLapackBackend::name() const noexcept {
    return "blas-lapack";
}

BackendCapabilities BlasLapackBackend::capabilities() const noexcept {
    return BackendCapabilities{.family = BackendFamily::LocalHost, .thread_safe = true, .uses_internal_threads = true, .distributed_mpi = false, .uses_device_memory = false};
}

MatrixComplex BlasLapackBackend::inverse(MatrixComplex a) const {
    if (a.rows() != a.cols()) {
        throw std::runtime_error("BlasLapackBackend::inverse: matrix must be square");
    }

    const int n = checked_lapack_int(a.rows(), "matrix dimension");
    const int lda = n;
    std::vector<int> ipiv(static_cast<std::size_t>(n));

    int info = 0;
    zgetrf_(&n, &n, a.data().data(), &lda, ipiv.data(), &info);
    if (info < 0) {
        throw std::runtime_error("zgetrf_: argument " + std::to_string(-info) + " had an illegal value");
    }
    if (info > 0) {
        throw std::runtime_error("zgetrf_: matrix is singular at U(" + std::to_string(info) + "," + std::to_string(info) + ")");
    }
    validate_lu_pivots(ipiv, n);

    MatrixComplex inv(static_cast<std::size_t>(n), static_cast<std::size_t>(n), Complex{0.0, 0.0});
    for (int i = 0; i < n; ++i) {
        inv(static_cast<std::size_t>(i), static_cast<std::size_t>(i)) = Complex{1.0, 0.0};
    }

    // MatrixComplex is row-major.  Fortran LAPACK sees the same buffer as the
    // transpose, so solving A^T X = I in column-major storage leaves A^{-1}
    // in the original row-major layout.
    const char trans = 'N';
    zgetrs_(&trans, &n, &n, a.data().data(), &lda, ipiv.data(), inv.data().data(), &n, &info);
    if (info < 0) {
        throw std::runtime_error("zgetrs_: argument " + std::to_string(-info) + " had an illegal value");
    }

    return inv;
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

    // Row-major C = op(A) op(B) is represented to Fortran BLAS as
    // C^T = op(B)^T op(A)^T over the same buffers.
    const char transa = to_fortran_transpose(trans_b);
    const char transb = to_fortran_transpose(trans_a);
    const int fm = checked_blas_int(n, "m");
    const int fn = checked_blas_int(m, "n");
    const int fk = checked_blas_int(k_a, "k");
    const int lda = checked_blas_int(b.cols(), "lda");
    const int ldb = checked_blas_int(a.cols(), "ldb");
    const int ldc = checked_blas_int(c.cols(), "ldc");
    zgemm_(&transa,
           &transb,
           &fm,
           &fn,
           &fk,
           &alpha,
           b.data().data(),
           &lda,
           a.data().data(),
           &ldb,
           &beta,
           c.data().data(),
           &ldc);

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

    const int rows = checked_blas_int(a.rows(), "a.rows");
    const int cols = checked_blas_int(a.cols(), "a.cols");
    const int lda = cols;
    const int inc = 1;

    if (trans_a == MatrixTranspose::NoTranspose) {
        const char trans = 'T';
        zgemv_(&trans, &cols, &rows, &alpha, a.data().data(), &lda, x_complex.data(), &inc, &beta, y.data(), &inc);
    } else {
        const char trans = 'N';
        zgemv_(&trans, &cols, &rows, &alpha, a.data().data(), &lda, x_complex.data(), &inc, &beta, y.data(), &inc);
        if (trans_a == MatrixTranspose::ConjugateTranspose) {
            std::transform(y.begin(), y.end(), y.begin(), [](Complex value) { return std::conj(value); });
        }
    }

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
