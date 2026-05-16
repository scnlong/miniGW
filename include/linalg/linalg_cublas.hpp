#pragma once

#include "linalg/linalg.hpp"

#include <cstddef>
#include <memory>

namespace gw::linalg {

// Generic CUDA backend using cuBLAS/cuSolver.
//
// This class implements the gw::linalg::Backend interface with host-wrapper
// semantics: inputs and outputs are still replicated host MatrixComplex
// objects.  Each call stages data to the GPU, invokes cuBLAS/cuSolver, and
// copies the result back.
//
// In the main GW executable, selecting --linalg-backend cublas creates this
// backend through the backend factory.  Its capabilities identify it as a
// device-memory backend, after which the GW driver switches to the dedicated
// DeviceScreeningWorkspace path for GPU-resident V_ph/epsilon/W_c matrices and
// contraction panels.  The backend itself is not a distributed-MPI matrix
// backend; MPI+CUDA parallelism is implemented outside this class by assigning
// different frequency points to different ranks.
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
