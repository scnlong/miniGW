#include "gw/linalg_scalapack.hpp"

#include <stdexcept>

namespace gw::linalg {
namespace {
[[noreturn]] void not_implemented() {
    throw std::runtime_error("ScaLAPACK backend interface is present, but the distributed matrix implementation is not implemented yet. Add a DistributedMatrix abstraction and ScaLAPACK pzgetrf/pzgetri or solve path before using this backend.");
}
} // namespace

std::string_view ScalapackBackend::name() const noexcept { return "scalapack-placeholder"; }
BackendCapabilities ScalapackBackend::capabilities() const noexcept {
    return BackendCapabilities{.thread_safe = true, .uses_internal_threads = false, .distributed_mpi = true, .uses_device_memory = false};
}
MatrixComplex ScalapackBackend::inverse(MatrixComplex) const { not_implemented(); }
MatrixComplex ScalapackBackend::gemm(const MatrixComplex&, const MatrixComplex&, MatrixTranspose, MatrixTranspose) const { not_implemented(); }
std::vector<Complex> ScalapackBackend::gemv(const MatrixComplex&, const std::vector<double>&, MatrixTranspose) const { not_implemented(); }
Complex ScalapackBackend::quadratic_form(const std::vector<double>&, const MatrixComplex&) const { not_implemented(); }
std::shared_ptr<const Backend> make_scalapack_backend() { static const auto backend = std::make_shared<ScalapackBackend>(); return backend; }

} // namespace gw::linalg
