# Linear-algebra backend interface

miniGW separates the GW algorithm from the dense linear-algebra implementation through the `gw::linalg::Backend` interface.  This document describes the current boundary and the intended responsibilities of each backend family.

The interface is declared in:

```text
include/linalg/linalg.hpp
```

The reference implementation is in:

```text
src/linalg/linalg.cpp
```

Backend construction and validation are handled by:

```text
include/linalg/backend_factory.hpp
src/linalg/backend_factory.cpp
```

## Interface

The common interface is:

```cpp
namespace gw::linalg {

class Backend {
public:
    virtual ~Backend() = default;

    virtual std::string_view name() const noexcept = 0;
    virtual BackendCapabilities capabilities() const noexcept;

    virtual MatrixComplex inverse(MatrixComplex a) const = 0;

    virtual MatrixComplex gemm(const MatrixComplex& a,
                               const MatrixComplex& b,
                               MatrixTranspose trans_a,
                               MatrixTranspose trans_b) const = 0;

    virtual std::vector<Complex> gemv(const MatrixComplex& a,
                                      const std::vector<double>& x,
                                      MatrixTranspose trans_a) const = 0;

    virtual Complex quadratic_form(const std::vector<double>& x,
                                   const MatrixComplex& a) const = 0;
};

} // namespace gw::linalg
```

The interface is intentionally small.  It covers only the operations currently needed by the host-style GW path:

- `inverse(MatrixComplex)` for dielectric and Coulomb matrix inversions;
- `gemm(A,B,trans_a,trans_b)` for dense matrix-matrix products such as `(epsilon^{-1}-I)^T * inv(V_ph)` and `W_c * X`;
- `gemv(A,x,trans_a)` for matrix-vector products;
- `quadratic_form(x,A)` for scalar contractions of the form `x^T A x`.

This interface is not sufficient by itself for high-performance distributed or GPU-resident GW.  It is a compatibility boundary for local and wrapper-style backends; the production ScaLAPACK, COSMA, and CUDA paths also use specialized workspace and matrix-ownership classes.

## Backend capabilities

`BackendCapabilities` describes how a backend may be used by the high-level driver:

```cpp
struct BackendCapabilities {
    BackendFamily family;
    bool thread_safe;
    bool uses_internal_threads;
    bool distributed_mpi;
    bool uses_device_memory;
    bool uses_cosma_pxgemm;
    bool external_provider_may_use_gpu;
};
```

The most important flags are:

- `distributed_mpi`: multiple MPI ranks collectively own and operate on distributed matrices inside the backend path. This flag is reserved for ScaLAPACK/COSMA-style distributed screening paths.
- `uses_device_memory`: the backend uses GPU/device memory internally or selects a GPU-resident execution path. This does not imply distributed multi-GPU ownership.
- `uses_cosma_pxgemm`: distributed GEMM is routed explicitly to COSMA's prefixed PBLAS-compatible provider, `cosma_pzgemm_`.
- `external_provider_may_use_gpu`: the external provider may use GPUs internally.  This is diagnostic and does not imply that miniGW owns COSMA's device buffers.

The current cuBLAS backend has `uses_device_memory=true` and `distributed_mpi=false`. MPI is used outside the backend to distribute frequency points, while each rank owns its own CUDA screening workspace. Setting `distributed_mpi=true` would require a different backend that defines distributed GPU matrix ownership and collective multi-rank operations.

These flags are used to select the correct screening workspace and to reject incompatible execution modes.

## Backend families

### Reference backend

Files:

```text
include/linalg/linalg.hpp
src/linalg/linalg.cpp
```

The reference backend is self-contained and intended for correctness and portability.  It should remain simple and easy to audit.  It is not a performance backend.

### BLAS/LAPACK backend

Files:

```text
include/linalg/linalg_blas_lapack.hpp
src/linalg/linalg_blas_lapack.cpp
```

The BLAS/LAPACK backend operates on replicated host `MatrixComplex` objects.  It uses:

- Fortran BLAS `zgemm_` and `zgemv_` for dense GEMM/GEMV;
- Fortran LAPACK `zgetrf_` and `zgetrs_` for LU factorization and inversion/solve operations.

Because miniGW stores matrices in row-major order, the wrapper must translate transpose flags, multiplication order, and leading dimensions carefully.  The implementation calls the Fortran BLAS/LAPACK ABI directly instead of linking CBLAS or `liblapacke`, so dense kernels and LU/solve routines come from the same provider selected by CMake.

### ScaLAPACK backend

Files:

```text
include/linalg/linalg_scalapack.hpp
src/linalg/linalg_scalapack.cpp
include/matrix/distributed_matrix.hpp
src/matrix/distributed_matrix.cpp
include/workspace/distributed_screening_workspace.hpp
src/workspace/distributed_screening_workspace.cpp
```

The ScaLAPACK support has two layers:

1. A replicated-wrapper backend implementing the generic `gw::linalg::Backend` operations by scattering replicated host matrices to BLACS block-cyclic matrices, calling ScaLAPACK/PBLAS, and gathering results back.
2. A GW-specific distributed screening workspace that keeps `V_ph`, `epsilon`, `inv(V_ph)`, and `W_c` as distributed matrices during the frequency loop.

The second layer is the important one for memory scaling.  It avoids gathering `W_c` during the self-energy contraction and evaluates panel contractions through distributed GEMM.

### COSMA backend

Files:

```text
include/linalg/linalg_cosma.hpp
src/linalg/linalg_cosma.cpp
include/matrix/cosma_distributed_matrix.hpp
src/matrix/cosma_distributed_matrix.cpp
include/workspace/cosma_distributed_screening_workspace.hpp
src/workspace/cosma_distributed_screening_workspace.cpp
```

The COSMA path is not a CPU ScaLAPACK replacement in miniGW.  It is the explicit COSMA distributed-GEMM path intended for multi-node/multi-GPU screening runs. COSMA should not be interpreted as an automatic BLACS-grid tuner: miniGW still defines frequency groups and distributed matrix ownership, while COSMA optimizes GEMM communication and execution inside that context.

miniGW calls COSMA through the prefixed PBLAS-compatible ABI symbol:

```text
cosma_pzgemm_
```

rather than relying on ordinary `pzgemm_` symbol interposition.  This makes the backend choice explicit at the source-code level.

COSMA is used for distributed GEMM operations, including:

```text
W_c = (epsilon^{-1} - I)^T * inv(V_ph)
Y   = W_c X
```

ScaLAPACK/BLACS infrastructure is still required for descriptors, process grids, and distributed factorization/solve operations.  In particular, COSMA does not replace the dielectric inversion path in this code; it replaces the distributed GEMM provider.

In a GPU-enabled COSMA stack, COSMA may use Tiled-MM and CUDA/NCCL internally.  miniGW does not manage COSMA's internal GPU buffers directly; actual GPU execution should be verified with runtime tools such as `nvidia-smi`, Nsight Systems, or COSMA's own diagnostics.

### CUDA cuBLAS/cuSolver backend

Files:

```text
include/linalg/linalg_cublas.hpp
src/linalg/linalg_cublas.cu
include/workspace/device_screening_workspace.hpp
src/workspace/device_screening_workspace.cu
include/workspace/device_pq_ph_panel.hpp
src/workspace/device_pq_ph_panel.cu
```

There are two CUDA-related layers:

1. A lower-level cuBLAS/cuSolver backend that implements generic backend operations with host-wrapper semantics. It remains useful as a factory-created `gw::linalg::Backend` object and as a capability carrier for `--linalg-backend cublas`.
2. A GW-specific `DeviceScreeningWorkspace` that keeps `V_ph`, `epsilon`, `W_c`, solver workspaces, and contraction panels device-resident across frequency points.

The second layer is the preferred CUDA path.  The lower-level host-wrapper API is useful for integration and correctness testing, but hiding host-device copies inside every small backend call is not the desired high-performance design. The cuBLAS path is per-rank: it does not distribute one screening matrix across multiple GPUs.

## Execution-policy compatibility

The backend factory checks the selected backend against the requested execution policy.

General rules:

- Local host backends may use serial execution, OpenMP kernel loops, or MPI frequency distribution.
- OpenMP frequency parallelism requires a thread-safe backend.
- ScaLAPACK and COSMA distributed screening require MPI and do not support OpenMP frequency parallelism.
- CUDA device-resident screening supports single-rank serial execution and MPI frequency distribution. With multiple MPI ranks, CUDA runs should use `--frequency-parallel mpi`.
- Multi-rank serial frequency execution is reserved for collective distributed backends such as ScaLAPACK/COSMA, where all ranks in a group cooperate on the same frequency point.
- Input ERI ownership is still replicated on the host in all current backends.

## Current command-line backend names

The executable accepts:

```text
--linalg-backend reference
--linalg-backend blas-lapack
--linalg-backend scalapack
--linalg-backend cosma
--linalg-backend cublas
```

The current command-line default is:

```text
blas-lapack
```

provided the executable was built with `GW_ENABLE_BLAS_LAPACK=ON`.  The reference backend remains the fallback and minimal correctness backend.

## Design boundary

The high-level GW driver should not contain backend-specific dense linear-algebra kernels.  It should express operations in terms of:

- screening workspaces;
- panel views;
- backend GEMM/inversion calls;
- explicit distributed or device matrix ownership classes when needed.

This boundary keeps the physics-level workflow readable while allowing backend implementations to specialize memory ownership, communication, and device execution.

## CUDA rank-to-GPU mapping

For CUDA runs, `--tasks-per-gpu N` controls local-rank-to-GPU mapping with the formula

```text
device = (local_rank / tasks_per_gpu) % visible_device_count
```

It groups `N` consecutive local MPI ranks onto one visible GPU, then cycles over visible GPUs. It is not a hard limit on the total number of ranks that may use a GPU. For balanced GPU sharing, choose the number of local ranks as a multiple of `visible_device_count * N`.

For example, with two visible GPUs and `--tasks-per-gpu 2`:

```text
4 local ranks:
  ranks 0,1 -> GPU 0
  ranks 2,3 -> GPU 1

10 local ranks:
  ranks 0,1,4,5,8,9 -> GPU 0
  ranks 2,3,6,7     -> GPU 1
```

## Possible Kokkos direction

Kokkos is a possible future rank-local execution layer, not a replacement for ScaLAPACK or COSMA. In the distributed backends, ScaLAPACK/COSMA would still own the distributed matrix problem. Kokkos could be used inside each rank for local panel assembly, host/device staging, custom kernels, pinned-buffer management, and CPU/GPU pipeline experiments.

For the per-rank cuBLAS path, Kokkos could replace some raw CUDA kernels and manual memory bookkeeping while retaining cuBLAS/cuSolver for large dense GEMM and solve operations.

## Known limitations

- The generic `gw::linalg::Backend` interface is too narrow for fully optimized distributed or GPU-resident workflows; specialized workspaces are still required.
- ScaLAPACK distributes the dominant CPU `nph x nph` screening matrices but does not distribute the full ERI input.
- COSMA is currently used only as the explicit distributed GEMM provider; ScaLAPACK still handles distributed inversion/solve infrastructure.
- CUDA/cuBLAS is a per-rank GPU-resident path with MPI frequency distribution, not a distributed multi-GPU matrix backend.
- CUDA support still starts from replicated host input data.
- A future production design should introduce distributed/tiled integral ownership and a stronger separation between host, distributed, and device matrix types.
