#include "gw/linalg_cosma.hpp"

#include <stdexcept>

namespace gw::linalg {
namespace {
[[noreturn]] void not_implemented() {
    throw std::runtime_error("COSMA backend interface is present, but this placeholder cannot replace the full GW dense-linear-algebra path. COSMA should be integrated as a distributed GEMM engine behind a distributed matrix backend, typically together with ScaLAPACK/SLATE for solve/factorization.");
}
} // namespace

std::string_view CosmaBackend::name() const noexcept { return "cosma-placeholder"; }
BackendCapabilities CosmaBackend::capabilities() const noexcept {
    return BackendCapabilities{.thread_safe = true, .uses_internal_threads = false, .distributed_mpi = true, .uses_device_memory = false};
}
MatrixComplex CosmaBackend::inverse(MatrixComplex) const { not_implemented(); }
MatrixComplex CosmaBackend::gemm(const MatrixComplex&, const MatrixComplex&, MatrixTranspose, MatrixTranspose) const { not_implemented(); }
std::vector<Complex> CosmaBackend::gemv(const MatrixComplex&, const std::vector<double>&, MatrixTranspose) const { not_implemented(); }
Complex CosmaBackend::quadratic_form(const std::vector<double>&, const MatrixComplex&) const { not_implemented(); }
std::shared_ptr<const Backend> make_cosma_backend() { static const auto backend = std::make_shared<CosmaBackend>(); return backend; }

} // namespace gw::linalg
