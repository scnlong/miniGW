#pragma once

#include "gw/linalg.hpp"

#include <memory>

namespace gw::linalg {

// Interface-compatible ScaLAPACK backend.
//
// This class keeps the legacy replicated MatrixComplex Backend interface for
// integration tests and non-GW call sites.  The production ScaLAPACK GW path
// uses matrix::DistributedMatrixComplex and DistributedScreeningWorkspace
// directly, so V_ph, epsilon, inv(V_ph), and W_c are block-cyclic and are not
// gathered inside the screening loop.
class ScalapackBackend final : public Backend {
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

// COSMA pxgemm backend.  Factorization/solve still use ScaLAPACK, but
// distributed GEMM calls are routed explicitly to COSMA's prefixed PBLAS ABI
// symbol, e.g. cosma_pzgemm_.  This is intentionally different from the
// ScaLAPACK backend, which calls the ordinary pzgemm_ symbol.
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

private:
    ScalapackBackend scalapack_{};
};

[[nodiscard]] std::shared_ptr<const Backend> make_scalapack_backend();
[[nodiscard]] std::shared_ptr<const Backend> make_cosma_pxgemm_backend();

} // namespace gw::linalg
