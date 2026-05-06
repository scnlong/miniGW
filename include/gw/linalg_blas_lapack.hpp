#pragma once

#include "gw/linalg.hpp"

#include <memory>
#include <string_view>

namespace gw::linalg {

class BlasLapackBackend final : public Backend {
public:
    [[nodiscard]] std::string_view name() const noexcept override;

    [[nodiscard]] MatrixComplex inverse(MatrixComplex a) const override;

    [[nodiscard]] MatrixComplex gemm(const MatrixComplex& a,
                                     const MatrixComplex& b,
                                     MatrixTranspose trans_a = MatrixTranspose::NoTranspose,
                                     MatrixTranspose trans_b = MatrixTranspose::NoTranspose) const override;

    [[nodiscard]] std::vector<Complex> gemv(const MatrixComplex& a,
                                            const std::vector<double>& x,
                                            MatrixTranspose trans_a = MatrixTranspose::NoTranspose) const override;

    [[nodiscard]] Complex quadratic_form(const std::vector<double>& x,
                                         const MatrixComplex& a) const override;
};

[[nodiscard]] std::shared_ptr<const Backend> make_blas_lapack_backend();

} // namespace gw::linalg
