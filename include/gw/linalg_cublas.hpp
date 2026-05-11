#pragma once

#include "gw/linalg.hpp"

#include <cstddef>
#include <memory>

namespace gw::linalg {

// Single-rank CUDA backend using cuBLAS/cuSolver.
//
// This is an interface-compatible host wrapper: inputs and outputs are still
// replicated MatrixComplex objects.  The backend copies matrices to the GPU,
// calls cuBLAS/cuSolver, and copies results back.  It is a real CUDA call path
// intended for correctness and incremental integration.  A high-performance GW
// path should keep V_ph/epsilon/workspaces device-resident across frequencies.
class CublasBackend final : public Backend {
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

[[nodiscard]] std::shared_ptr<const Backend> make_cublas_backend();

[[nodiscard]] int cuda_device_count();
[[nodiscard]] int select_cuda_device_for_local_rank(std::size_t mpi_local_rank, std::size_t tasks_per_gpu);
int set_cuda_device_for_local_rank(std::size_t mpi_local_rank, std::size_t tasks_per_gpu);

} // namespace gw::linalg
