#pragma once

#include "gw/linalg.hpp"

#include <memory>

namespace gw::linalg {

// Explicit COSMA backend.
//
// This backend is intentionally separated from the ScaLAPACK backend.  It keeps
// ScaLAPACK for distributed factorization/solve and BLACS descriptors, but all
// distributed GEMM calls are routed to COSMA's prefixed PBLAS ABI symbol
// cosma_pzgemm_.  Ordinary ScaLAPACK code never calls COSMA symbols.
class CosmaPxgemmBackend final : public Backend {
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

[[nodiscard]] std::shared_ptr<const Backend> make_cosma_pxgemm_backend();

} // namespace gw::linalg
