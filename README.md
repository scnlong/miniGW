[![Build and Regression](https://github.com/scnlong/miniGW/actions/workflows/build-regression.yml/badge.svg?branch=master)](https://github.com/scnlong/miniGW/actions/workflows/build-regression.yml)

<p align="center">
  <img src="docs/logo/miniGW-logo.png" alt="miniGW logo" width="350">
</p>

# miniGW

miniGW is a compact molecular G0W0 code written in modern C++. It uses PySCF to generate the DFT starting-point data and reads the resulting molecular-orbital quantities from a single HDF5 input file.

The code is intended as a development and experimentation platform for molecular GW workflows, linear-algebra backend integration, MPI frequency distribution, distributed ScaLAPACK/COSMA screening paths, and CUDA-accelerated GPU-resident screening workspaces.

Resolution of the Identity (RI) / density fitting is not implemented. The current input format requires a full four-index MO-basis ERI tensor.

The G0W0 method is a many-body perturbation theory approach in which quasiparticle excitation energies are obtained by evaluating the electronic self-energy Σ = iG0W0 from a non-interacting Green’s function G0 and a screened Coulomb interaction W0, typically starting from a density functional theory reference.

## Current scope

Implemented components include:

- transformed Gauss-Legendre and linear frequency grids;
- occupied-virtual particle-hole index mapping;
- bare exchange in the MO basis;
- diagonal non-interacting polarizability in the particle-hole basis;
- particle-hole Coulomb matrix construction;
- panel-based \(P_{pq,ia}\) contraction without materializing the full `pq_ph(nmo,nmo,nph)` tensor;
- correlation self-energy on the imaginary axis;
- continued-fraction Padé analytic continuation;
- iterative diagonal quasiparticle-energy update;
- HDF5 reader for the PySCF-generated single-file input bundle;
- local reference and BLAS/LAPACK linear-algebra backends;
- MPI frequency distribution;
- ScaLAPACK distributed screening path;
- COSMA GPU backend for multi-node multi-GPU distributed screening;
- CUDA cuBLAS/cuSolver backend with per-rank GPU-resident screening workspaces.

## Repository layout

```text
include/                 Public C++ headers
src/                     C++ and CUDA implementation files
cmake/                   CMake helper modules and regression-test registration
pyscf_prep/              PySCF input-generation scripts
regression_tests/        Reference H2O regression inputs and outputs
scripts/                 Local, CI, and cluster regression-test helpers
docs/                    Formulation and backend-interface notes
.github/workflows/       GitHub Actions CI workflow
```

Important source areas:

```text
include/gw/              GW data structures and algorithm declarations
include/linalg/          Linear-algebra backend interface and backend factories
include/workspace/       Host, distributed, and device screening workspaces
include/matrix/          Dense, distributed, and ownership-related matrix types
src/gw/gw.cpp            Main G0W0 workflow
src/hdf5_input.cpp       HDF5 input reader
src/linalg/              Reference, BLAS/LAPACK, ScaLAPACK, COSMA, and CUDA backend code
src/workspace/           Screening workspace implementations
```

## Dependencies

miniGW always requires:

- a C++20 compiler;
- CMake >= 3.21;
- HDF5 C development files.

Optional C++/HPC dependencies:

- OpenMP, if `GW_ENABLE_OPENMP=ON`;
- BLAS and LAPACK, if `GW_ENABLE_BLAS_LAPACK=ON`;
- MPI, if `GW_ENABLE_MPI=ON`;
- ScaLAPACK/BLACS, if `GW_ENABLE_SCALAPACK=ON`;
- COSMA with its GPU-enabled dependency stack, if `GW_ENABLE_COSMA=ON`;
- CUDA Toolkit, cuBLAS, and cuSolver, if `GW_ENABLE_CUDA=ON`.

Python dependencies are only needed for generating new PySCF input files:

- `numpy`;
- `h5py`;
- `pyscf`.

The C++ code does not require PySCF or `h5py` at build time if the HDF5 regression inputs already exist.

On Ubuntu, a typical CPU/MPI development environment can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  gfortran \
  pkg-config \
  python3 \
  libhdf5-dev \
  libopenblas-dev \
  liblapack-dev \
  libopenmpi-dev \
  openmpi-bin \
  libscalapack-openmpi-dev
```

## Build

A standard developer build using the repository CMake cache file is:

```bash
cmake -S . -B build -C cmake_install.cmake
cmake --build build -j 4
```

The current default configuration enables several optional CPU/MPI backends. For a minimal reference-only build, use explicit switches:

```bash
cmake -S . -B build-reference \
  -DCMAKE_BUILD_TYPE=Release \
  -DGW_ENABLE_TESTS=OFF \
  -DGW_ENABLE_OPENMP=OFF \
  -DGW_ENABLE_OPENMP_FREQUENCY_PARALLEL=OFF \
  -DGW_ENABLE_MPI=OFF \
  -DGW_ENABLE_BLAS_LAPACK=OFF \
  -DGW_ENABLE_SCALAPACK=OFF \
  -DGW_ENABLE_COSMA=OFF \
  -DGW_ENABLE_CUDA=OFF

cmake --build build-reference -j
```

When using this minimal build, run with the reference backend explicitly:

```bash
./build-reference/gw \
  --input-dir regression_tests/h2o_serial \
  --linalg-backend reference \
  --frequency-parallel serial
```

## Input format

Run the PySCF preparation script to generate the miniGW input bundle:

```bash
python3 pyscf_prep/pyscf_g0w0_prep.py
```

miniGW expects the input directory to contain:

```text
gw_input.h5
```

The HDF5 file contains the datasets:

```text
/mo_energy   shape: (nmo,)                float64
/eri_mo      shape: (nmo,nmo,nmo,nmo)     float64
/vxc_mo      shape: (nmo,nmo)             float64
```

and the file attributes:

```text
nocc          integer
fermi_energy  float64
```

Energies are in Hartree. Arrays are written as C-order `float64` data and are read directly into miniGW's row-major containers.

## Run

Basic run:

```bash
./build/gw \
  --input-dir regression_tests/h2o_serial \
  --freq-points 200 \
  --pade-params 16 \
  --state 5 \
  --output-dir gw_output
```

Useful options:

```text
--input-dir PATH              Directory containing gw_input.h5
--output-dir PATH             Directory for E_c_before_Pade.out, E_c.out, and gw.out
--freq-points N               Number of imaginary-frequency points
--pade-params N               Number of Padé parameters
--state N                     1-based orbital index
--all-states                  Compute all diagonal states
--linalg-backend NAME         reference, blas-lapack, scalapack, cosma, or cublas
--frequency-parallel MODE     auto, serial, mpi, or openmp
--kernel-parallel MODE        auto, serial, or openmp
--contraction-panel-size N    Number of (p,k) vectors per Sigma_c contraction panel
--scalapack-ranks-per-group N MPI ranks per distributed frequency group
--tasks-per-gpu N             Local-rank-to-GPU mapping block size for CUDA runs
```

See the complete command-line interface with:

```bash
./build/gw --help
```

## Regression tests

Regression tests are registered through CTest when `GW_ENABLE_TESTS=ON`.

To list available tests:

```bash
ctest --test-dir build -N
```

To run all registered tests:

```bash
ctest --test-dir build --output-on-failure
```

For the standard local build-and-regression workflow, run from the repository root:

```bash
bash scripts/regression_tests_local.sh
```

The regression script builds the code, runs selected serial, BLAS/LAPACK, MPI, and ScaLAPACK H2O tests, and writes a log file:

```text
regression_tests.log
```

The script uses `set -euo pipefail`, so any failed `ctest` command returns a non-zero exit code. This is the mechanism used by GitHub Actions to mark the build-and-regression workflow as failed.

For GitHub CI with limited runner resources, the number of MPI ranks can be overridden with an environment variable if the script is written to use it:

```bash
GW_TEST_MPI_RANKS=4 bash scripts/regression_tests_local.sh
```

## Linear-algebra backends

miniGW selects the dense linear-algebra implementation through `gw::linalg::Backend`.

The command-line option is:

```bash
--linalg-backend reference
--linalg-backend blas-lapack
--linalg-backend scalapack
--linalg-backend cosma
--linalg-backend cublas
```

The command-line default is currently:

```text
blas-lapack
```

when the executable was built with `GW_ENABLE_BLAS_LAPACK=ON`.

The backend families target different scaling regimes:

- `reference` and `blas-lapack` are replicated host-memory paths.
- `scalapack` distributes the dominant CPU screening matrices inside MPI frequency groups.
- `cosma` targets distributed multi-rank/multi-GPU GEMM in the screening workflow when linked against a GPU-enabled COSMA stack.
- `cublas` is a per-rank CUDA path: each MPI rank owns its own GPU-resident screening workspace, and MPI distributes frequency points across ranks. It does not distribute one screening matrix across multiple GPUs.

In all current paths, the four-index ERI input remains replicated in host memory.

### Reference backend

The reference backend is self-contained and intended for correctness testing and portability. It is not optimized for production performance.

Example:

```bash
./build/gw \
  --input-dir regression_tests/h2o_serial \
  --linalg-backend reference \
  --frequency-parallel serial
```

### BLAS/LAPACK backend

`--linalg-backend blas-lapack` uses the BLAS/LAPACK provider selected by CMake for replicated host matrices:

- Fortran BLAS `zgemm_` for dense complex GEMM;
- Fortran BLAS `zgemv_` for dense complex GEMV;
- Fortran LAPACK `zgetrf_` and `zgetrs_` to obtain inverse matrices by solving against the identity, avoiding both the more fragile `zgetri` path and mixed LAPACKE/provider runtime linkage.

The backend intentionally does not link CBLAS or `liblapacke`. This keeps GEMM, GEMV, and LU/solve on the same BLAS/LAPACK implementation chosen by CMake, for example OpenBLAS or Intel oneMKL through `BLA_VENDOR`.

Configure with:

```bash
cmake -S . -B build-blas \
  -DCMAKE_BUILD_TYPE=Release \
  -DGW_ENABLE_BLAS_LAPACK=ON \
  -DBLA_VENDOR=OpenBLAS

cmake --build build-blas -j
```

### MPI frequency distribution

MPI frequency distribution assigns different imaginary-frequency points to different MPI ranks and reduces the final self-energy columns.

Example:

```bash
mpirun -np 4 ./build/gw \
  --input-dir regression_tests/h2o_mpi \
  --frequency-parallel mpi \
  --linalg-backend blas-lapack
```

### ScaLAPACK backend

`--linalg-backend scalapack` enables a CPU distributed screening path. In this path, the dominant screening matrices such as `V_ph`, the left dielectric matrix `I - diag(Pi0) V_ph`, and `W_c` are represented as BLACS block-cyclic distributed matrices inside `DistributedScreeningWorkspace`. The code does not explicitly form `inv(V_ph)` in this path; it computes the correlation screened interaction as `W_c = (I - diag(Pi0) V_ph)^(-1) diag(Pi0)`. The four-index ERI tensor remains replicated on each MPI rank; ScaLAPACK is used for the screening linear algebra, not for distributed integral storage.

Configure with:

```bash
cmake -S . -B build-scalapack \
  -DCMAKE_BUILD_TYPE=Release \
  -DGW_ENABLE_MPI=ON \
  -DGW_ENABLE_SCALAPACK=ON \
  -DSCALAPACK_LIBRARIES="/path/to/libscalapack.so"

cmake --build build-scalapack -j
```

If `SCALAPACK_LIBRARIES` is omitted, CMake searches for common ScaLAPACK library names such as `scalapack`, `scalapack-openmpi`, or `scalapack-mpich`.

All ranks can cooperate on every frequency point:

```bash
mpirun -np 4 ./build-scalapack/gw \
  --input-dir regression_tests/h2o_scalapack \
  --frequency-parallel serial \
  --linalg-backend scalapack
```

Frequency batching with independent ScaLAPACK communicator groups is also supported:

```bash
mpirun -np 16 ./build-scalapack/gw \
  --input-dir regression_tests/h2o_scalapack \
  --frequency-parallel mpi \
  --scalapack-ranks-per-group 4 \
  --linalg-backend scalapack
```

In grouped mode, `MPI_COMM_WORLD` is split into frequency groups. Each group owns a BLACS/ScaLAPACK grid and processes one frequency point at a time; different groups process different frequency indices. The option `--scalapack-ranks-per-group` controls the number of ranks in each distributed frequency group. Despite its current name, this grouping concept is also reused by the COSMA backend.

The ERI tensor is still replicated in host memory. The current distributed path removes replicated ownership of the dominant `nph x nph` screening matrices, but a full distributed-memory GW implementation would still require distributed or tiled integral storage and contraction.

### COSMA GPU backend

In miniGW, `--linalg-backend cosma` is a GPU-oriented COSMA backend. It is not used as a CPU replacement for the ScaLAPACK backend. The CPU distributed-memory path is the ScaLAPACK backend; the COSMA backend is intended for multi-node multi-GPU distributed screening when miniGW is built against a GPU-enabled COSMA stack. COSMA should be understood as a communication-optimized distributed GEMM provider, not as an automatic BLACS-grid tuner or a complete ScaLAPACK replacement.

Configure after loading suitable MPI, CUDA, COSMA, and required COSMA dependency modules:

```bash
cmake -S . -B build-cosma \
  -DCMAKE_BUILD_TYPE=Release \
  -DGW_ENABLE_MPI=ON \
  -DGW_ENABLE_COSMA=ON \
  -DGW_ENABLE_CUDA=ON \
  -DCOSMA_ROOT=/path/to/cosma

cmake --build build-cosma -j
```

Run with multiple MPI ranks and visible GPUs, for example:

```bash
mpirun -np 8 ./build-cosma/gw \
  --input-dir regression_tests/h2o_cosma \
  --frequency-parallel mpi \
  --linalg-backend cosma
```

For larger runs, the backend is intended to operate across multiple nodes and multiple GPUs, subject to the MPI launcher, GPU visibility, and the COSMA installation used on the target machine. The exact rank-to-GPU mapping and performance characteristics should be validated on the target cluster rather than inferred from the CPU ScaLAPACK backend. COSMA optimizes the distributed GEMM execution inside the distributed screening context; miniGW still controls frequency grouping, matrix ownership, and the replicated ERI input.

### CUDA cuBLAS/cuSolver backend

`--linalg-backend cublas` enables CUDA support when configured with `GW_ENABLE_CUDA=ON`.

The lower-level CUDA backend provides cuBLAS/cuSolver wrappers for replicated host matrices with host-wrapper semantics. The main GW CUDA path uses a dedicated `DeviceScreeningWorkspace`, where `V_ph`, `epsilon`, `inv(epsilon)-I`, `W_c`, solver workspaces, and contraction panel buffers are allocated once and reused on the GPU. In MPI mode, each rank owns its own CUDA workspace and processes a subset of the frequency points.

Configure with:

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGW_ENABLE_CUDA=ON

cmake --build build-cuda -j
```

Single-rank CUDA run:

```bash
./build-cuda/gw \
  --input-dir regression_tests/h2o_cublas_serial \
  --linalg-backend cublas \
  --frequency-parallel serial
```

For CUDA with multiple MPI ranks, use `--frequency-parallel mpi`. Multi-rank `--frequency-parallel serial` is reserved for collective distributed backends such as ScaLAPACK/COSMA, where all ranks in a group cooperate on the same frequency point. The cuBLAS path is not such a collective backend.

MPI frequency distribution with CUDA is supported:

```bash
mpirun -np 8 ./build-cuda/gw \
  --input-dir regression_tests/h2o_cublas_mpi \
  --linalg-backend cublas \
  --frequency-parallel mpi \
  --tasks-per-gpu 4
```

The mapping is local-rank based:

```text
device = (local_rank / tasks_per_gpu) % visible_device_count
```

`--tasks-per-gpu N` groups `N` consecutive local MPI ranks onto one visible GPU, then cycles over the visible GPUs. It does not cap the total number of ranks launched. For balanced GPU sharing, choose the number of local ranks as a multiple of `visible_device_count * N`. For example, with two visible GPUs and `--tasks-per-gpu 2`, four local ranks map as `0,1 -> GPU 0` and `2,3 -> GPU 1`; ten local ranks map as `0,1,4,5,8,9 -> GPU 0` and `2,3,6,7 -> GPU 1`.

The input ERI tensor remains replicated in host memory. CUDA contraction panels are provided by `DevicePqPhPanelView`, which either keeps the full ERI tensor resident on the GPU or streams only the requested panel through pinned host staging, depending on the selected storage mode and available device memory.

## Architecture notes

The code separates several concerns that should remain independent as the project grows:

- `include/execution.hpp` and `src/execution.cpp` define frequency-level execution policy: serial, OpenMP, or MPI.
- `include/matrix/ownership.hpp` records whether data are replicated host arrays, distributed block-cyclic arrays, or future device-resident arrays.
- `include/workspace/screening_workspace.hpp`, `include/workspace/distributed_screening_workspace.hpp`, and `include/workspace/device_screening_workspace.hpp` own the dominant GW screening temporaries.
- `include/workspace/pq_ph_panel.hpp` and `include/workspace/device_pq_ph_panel.hpp` provide panel views for the self-energy contraction without materializing the full `pq_ph` tensor.
- `include/linalg/backend_factory.hpp` and `src/linalg/backend_factory.cpp` select the requested backend and validate that it is compatible with the chosen execution mode.
- `src/gw/gw.cpp` contains the high-level G0W0 workflow and should remain independent of backend-specific implementation details where possible.

More detailed formulation and backend-interface notes are in:

```text
docs/formulation.md
docs/data_layout.md
docs/linalg_backend_interface.md
```

## Known limitations

- Resolution of the Identity (RI) / density fitting is not implemented.
- The input still requires a full four-index MO ERI tensor.
- The ERI tensor is replicated in host memory.
- The panel-based contraction avoids materializing the full `pq_ph(nmo,nmo,nph)` tensor, but does not yet solve the full distributed/tiled integral-storage problem.
- The ScaLAPACK path distributes the dominant CPU `nph x nph` screening matrices, but does not distribute the four-index ERI tensor.
- The COSMA path targets distributed GEMM in the screening workflow; it is not a complete replacement for ScaLAPACK factorization/solve or integral storage.
- The cuBLAS path is a per-rank GPU-resident path with MPI frequency distribution. It does not distribute one screening matrix across multiple GPUs.
- Multi-rank CUDA runs require `--frequency-parallel mpi`; multi-rank `--frequency-parallel serial` is reserved for collective distributed backends such as ScaLAPACK/COSMA.
