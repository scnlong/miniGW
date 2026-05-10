#include "gw/linalg_cosma.hpp"
#include "gw/linalg_scalapack.hpp"

#include <memory>
#include <stdexcept>

namespace gw::linalg {
namespace {

// See the README section generated with this patch.  COSMA has been used in
// several packaging/API variants.  Rather than hard-code a potentially wrong
// C++ symbol, this backend currently provides the correct semantic split:
// factorization/solve via ScaLAPACK and a single replacement point for GEMM.
// In an environment with a known COSMA API, replace this function by the COSMA
// multiply call and keep the rest of the backend unchanged.
[[nodiscard]] MatrixComplex cosma_or_scalapack_gemm(const MatrixComplex& a,
                                                    const MatrixComplex& b,
                                                    MatrixTranspose trans_a,
                                                    MatrixTranspose trans_b) {
    const ScalapackBackend scalapack;
    return scalapack.gemm(a, b, trans_a, trans_b);
}

} // namespace

std::string_view CosmaBackend::name() const noexcept {
    return "cosma-gemm-scalapack-solve-wrapper";
}

BackendCapabilities CosmaBackend::capabilities() const noexcept {
    return BackendCapabilities{.family = BackendFamily::DistributedHost,
                               .thread_safe = true,
                               .uses_internal_threads = false,
                               .distributed_mpi = true,
                               .uses_device_memory = false};
}

MatrixComplex CosmaBackend::inverse(MatrixComplex a) const {
    const ScalapackBackend scalapack;
    return scalapack.inverse(std::move(a));
}

MatrixComplex CosmaBackend::gemm(const MatrixComplex& a,
                                 const MatrixComplex& b,
                                 MatrixTranspose trans_a,
                                 MatrixTranspose trans_b) const {
    return cosma_or_scalapack_gemm(a, b, trans_a, trans_b);
}

std::vector<Complex> CosmaBackend::gemv(const MatrixComplex& a,
                                        const std::vector<double>& x,
                                        MatrixTranspose trans_a) const {
    const ScalapackBackend scalapack;
    return scalapack.gemv(a, x, trans_a);
}

Complex CosmaBackend::quadratic_form(const std::vector<double>& x, const MatrixComplex& a) const {
    const ScalapackBackend scalapack;
    return scalapack.quadratic_form(x, a);
}

std::shared_ptr<const Backend> make_cosma_backend() {
    static const std::shared_ptr<const Backend> backend = std::make_shared<CosmaBackend>();
    return backend;
}

} // namespace gw::linalg
