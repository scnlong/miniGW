#pragma once

#include "linalg/linalg.hpp"

#include <memory>

namespace gw::linalg {

// Interface-compatible ScaLAPACK backend.
//
// This class keeps the legacy replicated MatrixComplex Backend interface for
// integration tests and non-GW call sites.  The production ScaLAPACK GW path
// uses matrix::DistributedMatrixComplex and DistributedScreeningWorkspace
// directly, so V_ph, the left dielectric matrix, and W_c are block-cyclic and
// are not gathered inside the screening loop. The distributed screening path
// does not explicitly construct inv(V_ph).
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

[[nodiscard]] std::shared_ptr<const Backend> make_scalapack_backend();

} // namespace gw::linalg
