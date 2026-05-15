#include "linalg/linalg.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gw::linalg {
namespace {

[[nodiscard]] MatrixComplex identity_complex(std::size_t n) {
    MatrixComplex out(n, n, Complex{0.0, 0.0});
    for (std::size_t i = 0; i < n; ++i) {
        out(i, i) = Complex{1.0, 0.0};
    }
    return out;
}

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

} // namespace

std::string_view ReferenceBackend::name() const noexcept {
    return "reference-serial";
}

MatrixComplex ReferenceBackend::inverse(MatrixComplex a) const {
    const std::size_t n = a.rows();
    if (a.rows() != a.cols()) {
        throw std::runtime_error("linalg::inverse: matrix must be square");
    }
    MatrixComplex inv = identity_complex(n);

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        double pivot_abs = std::abs(a(col, col));
        for (std::size_t row = col + 1; row < n; ++row) {
            const double candidate = std::abs(a(row, col));
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot = row;
            }
        }
        if (pivot_abs < 1e-14) {
            throw std::runtime_error("linalg::inverse: near-singular matrix");
        }
        if (pivot != col) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(a(col, j), a(pivot, j));
                std::swap(inv(col, j), inv(pivot, j));
            }
        }

        const Complex diag = a(col, col);
        for (std::size_t j = 0; j < n; ++j) {
            a(col, j) /= diag;
            inv(col, j) /= diag;
        }
        for (std::size_t row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const Complex factor = a(row, col);
            if (std::abs(factor) == 0.0) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                a(row, j) -= factor * a(col, j);
                inv(row, j) -= factor * inv(col, j);
            }
        }
    }
    return inv;
}

MatrixComplex ReferenceBackend::gemm(const MatrixComplex& a,
                                     const MatrixComplex& b,
                                     MatrixTranspose trans_a,
                                     MatrixTranspose trans_b) const {
    const std::size_t m = effective_rows(a, trans_a);
    const std::size_t k_a = effective_cols(a, trans_a);
    const std::size_t k_b = effective_rows(b, trans_b);
    const std::size_t n = effective_cols(b, trans_b);
    if (k_a != k_b) {
        throw std::runtime_error("linalg::gemm: incompatible matrix shapes");
    }

    MatrixComplex c(m, n, Complex{0.0, 0.0});
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            Complex sum{0.0, 0.0};
            for (std::size_t k = 0; k < k_a; ++k) {
                sum += access(a, i, k, trans_a) * access(b, k, j, trans_b);
            }
            c(i, j) = sum;
        }
    }
    return c;
}

std::vector<Complex> ReferenceBackend::gemv(const MatrixComplex& a,
                                            const std::vector<double>& x,
                                            MatrixTranspose trans_a) const {
    const std::size_t m = effective_rows(a, trans_a);
    const std::size_t n = effective_cols(a, trans_a);
    if (x.size() != n) {
        throw std::runtime_error("linalg::gemv: incompatible matrix/vector shapes");
    }

    std::vector<Complex> y(m, Complex{0.0, 0.0});
    for (std::size_t i = 0; i < m; ++i) {
        Complex sum{0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) {
            sum += access(a, i, j, trans_a) * x[j];
        }
        y[i] = sum;
    }
    return y;
}

Complex ReferenceBackend::quadratic_form(const std::vector<double>& x, const MatrixComplex& a) const {
    if (a.rows() != a.cols() || x.size() != a.rows()) {
        throw std::runtime_error("linalg::quadratic_form: incompatible shapes");
    }

    const std::vector<Complex> ax = gemv(a, x);
    Complex value{0.0, 0.0};
    for (std::size_t i = 0; i < x.size(); ++i) {
        value += x[i] * ax[i];
    }
    return value;
}

const Backend& reference_backend() {
    static const ReferenceBackend backend;
    return backend;
}

std::shared_ptr<const Backend> make_reference_backend() {
    static const std::shared_ptr<const Backend> backend = std::make_shared<ReferenceBackend>();
    return backend;
}

} // namespace gw::linalg
