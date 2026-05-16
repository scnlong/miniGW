#include "linalg/backend_factory.hpp"

#include "execution.hpp"
#include "linalg/linalg.hpp"
#ifdef GW_HAS_BLAS_LAPACK_BACKEND
#include "linalg/linalg_blas_lapack.hpp"
#endif
#ifdef GW_HAS_SCALAPACK_BACKEND
#include "linalg/linalg_scalapack.hpp"
#endif
#ifdef GW_HAS_COSMA_BACKEND
#include "linalg/linalg_cosma.hpp"
#endif
#ifdef GW_HAS_CUDA_BACKEND
#include "linalg/linalg_cublas.hpp"
#endif

#include <stdexcept>

namespace gw {

std::shared_ptr<const linalg::Backend> make_local_linalg_backend(const Cli& cli) {
    if (cli.linalg_backend == "reference") {
        return linalg::make_reference_backend();
    }

    if (cli.linalg_backend == "blas-lapack") {
#ifdef GW_HAS_BLAS_LAPACK_BACKEND
        return linalg::make_blas_lapack_backend();
#else
        throw std::runtime_error("This executable was built without the BLAS/LAPACK backend. Reconfigure with -DGW_ENABLE_BLAS_LAPACK=ON, or use --linalg-backend reference.");
#endif
    }

    if (cli.linalg_backend == "scalapack") {
#ifdef GW_HAS_SCALAPACK_BACKEND
        return linalg::make_scalapack_backend();
#else
        throw std::runtime_error("This executable was built without the ScaLAPACK interface. Reconfigure with -DGW_ENABLE_SCALAPACK=ON.");
#endif
    }

    if (cli.linalg_backend == "cosma") {
#ifdef GW_HAS_COSMA_BACKEND
        // COSMA is used through its prefixed ScaLAPACK-compatible pxgemm ABI.
        // The distributed code path remains host/block-cyclic, but GEMM is
        // explicitly routed to cosma_pzgemm_ instead of ordinary pzgemm_.
        return linalg::make_cosma_pxgemm_backend();
#else
        throw std::runtime_error(
            "--linalg-backend cosma was requested, but this executable was not built "
            "with COSMA pxgemm support. Reconfigure with -DGW_ENABLE_SCALAPACK=ON "
            "-DGW_ENABLE_COSMA=ON and ensure libcosma_prefixed_pxgemm.a is available.");
#endif
    }

    if (cli.linalg_backend == "cublas") {
#ifdef GW_HAS_CUDA_BACKEND
        return linalg::make_cublas_backend();
#else
        throw std::runtime_error("This executable was built without the cuBLAS/cuSolver interface. Reconfigure with -DGW_ENABLE_CUDA=ON.");
#endif
    }

    throw std::runtime_error("Unknown --linalg-backend value: " + cli.linalg_backend + ". Supported values: reference, blas-lapack, scalapack, cosma, cublas.");
}

void validate_backend_for_execution(const linalg::Backend& backend, const ExecutionPolicy& execution) {
    const auto caps = backend.capabilities();

    if (execution.frequency_parallel_mode == FrequencyParallelMode::OpenMP) {
        if (!caps.thread_safe) {
            throw std::runtime_error("OpenMP frequency parallelism requires a thread-safe local backend.");
        }
        if (caps.distributed_mpi || caps.uses_device_memory) {
            throw std::runtime_error("OpenMP frequency parallelism is only supported for replicated local host backends. Use MPI frequency distribution or a backend-managed GPU queue for distributed/device backends.");
        }
    }

    if (caps.distributed_mpi) {
        if (execution.mpi_size == 1) {
            throw std::runtime_error("Distributed linear algebra backend selected, but MPI size is 1.");
        }
        if (execution.frequency_parallel_mode == FrequencyParallelMode::OpenMP) {
            throw std::runtime_error("Distributed ScaLAPACK/COSMA screening does not support OpenMP frequency parallelism. Use --frequency-parallel serial or mpi.");
        }
        if (execution.frequency_parallel_mode == FrequencyParallelMode::MPI && execution.frequency_group_size == 0) {
            throw std::runtime_error("Invalid ScaLAPACK communicator-group configuration.");
        }
        return;
    }

    if (execution.frequency_parallel_mode == FrequencyParallelMode::Serial && execution.mpi_size > 1) {
        if (caps.uses_device_memory) {
            throw std::runtime_error("Device backend selected with multiple MPI ranks in serial frequency mode. Current CUDA workspaces are per-rank and are not collective distributed-MPI matrix backends. Use --frequency-parallel mpi for MPI+CUDA; --tasks-per-gpu only controls local rank-to-GPU mapping.");
        }
        throw std::runtime_error("Replicated local host backend selected with multiple MPI ranks in serial frequency mode. Run without mpirun or use --frequency-parallel mpi.");
    }
}

} // namespace gw
