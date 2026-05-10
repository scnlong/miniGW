#pragma once

#include "gw/linalg.hpp"

#include <memory>

namespace gw::linalg {

// COSMA integration facade.
//
// COSMA is a distributed GEMM engine, not a full LAPACK replacement.  Therefore
// this backend is modeled as a ScaLAPACK-compatible distributed backend whose
// solve/inverse path is delegated to ScaLAPACK.  The current implementation keeps
// the replicated MatrixComplex interface and uses the ScaLAPACK path for all
// operations unless a project-specific COSMA GEMM adapter is added in
// linalg_cosma.cpp.
class CosmaBackend final : public Backend {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] BackendCapabilities capabilities() const noexcept override;

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

[[nodiscard]] std::shared_ptr<const Backend> make_cosma_backend();

} // namespace gw::linalg
