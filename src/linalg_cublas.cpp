#include "gw/linalg_cublas.hpp"

#include <stdexcept>

namespace gw::linalg {
namespace {
[[noreturn]] void not_implemented() {
    throw std::runtime_error("cuBLAS/cuSolver backend interface is present, but the device-resident workspace implementation is not implemented yet. Add CUDA buffers, cuBLAS/cuSolver handles, streams, and explicit host/device transfer policy before using this backend.");
}
} // namespace

std::string_view CublasBackend::name() const noexcept { return "cublas-placeholder"; }
BackendCapabilities CublasBackend::capabilities() const noexcept {
    return BackendCapabilities{.thread_safe = false, .uses_internal_threads = false, .distributed_mpi = false, .uses_device_memory = true};
}
MatrixComplex CublasBackend::inverse(MatrixComplex) const { not_implemented(); }
MatrixComplex CublasBackend::gemm(const MatrixComplex&, const MatrixComplex&, MatrixTranspose, MatrixTranspose) const { not_implemented(); }
std::vector<Complex> CublasBackend::gemv(const MatrixComplex&, const std::vector<double>&, MatrixTranspose) const { not_implemented(); }
Complex CublasBackend::quadratic_form(const std::vector<double>&, const MatrixComplex&) const { not_implemented(); }
std::shared_ptr<const Backend> make_cublas_backend() { static const auto backend = std::make_shared<CublasBackend>(); return backend; }

} // namespace gw::linalg
