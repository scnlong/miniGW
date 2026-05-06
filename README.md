# miniGW

A compact molecular G0W0 code written in modern C++20, using PySCF as the DFT starting point.

The project starts from a serial CPU implementation and a small self-contained `.npy` reader for importing PySCF-generated DFT data. The GW workflow calls dense linear algebra through `gw::linalg::Backend`, allowing BLAS/LAPACK, ScaLAPACK, COSMA, cuBLAS/cuSolver, or HIP backends to be added without rewriting the GW driver.

## Scope

Implemented modules:

- transformed Gauss-Legendre and linear frequency grids;
- particle-hole index mapping;
- bare exchange matrix in the MO basis;
- diagonal non-interacting polarizability in the particle-hole basis;
- particle-hole Coulomb matrix and projected `(p,q,ph)` tensor;
- correlation self-energy on the imaginary axis;
- continued-fraction Padé approximation;
- iterative diagonal quasiparticle-energy update;
- minimal NumPy `.npy` reader for little-endian `float64` arrays;
- explicit dense linear-algebra backend boundary with a serial reference backend.

## Build

```bash
cmake -S . -B build -C cmake_install.cmake
cmake --build build -j 4
```

## Expected input files

Run a pyscf DFT calculation (`pyscf_g0w0_prep.py` script from `pyscf_prep` directory) and its outputs contain:

```text
eri_mo.npy
mo_energy.npy
vxc_mo.npy
nocc.txt
fermi_energy.txt
```

The `.npy` reader currently supports C-order, little-endian `float64` arrays only.

## Run

```bash
./build/gw --input-dir /path/to/pyscf_output --freq-points 200 --pade-params 16 --state 5
```

Use `--all-states` to compute all diagonal states. 

## Regression Tests

```bash
ctest --test-dir build/ -N 
```

List all the cases to be tested.

```bash
ctest --test-dir build -j 4 --output-on-failure
```

Run all the regression test cases.


## Linear algebra backends

The default backend is `reference-serial`; it is intentionally simple and is meant for correctness and portability, not production performance. The current replacement boundary is documented in `docs/linalg_backend_interface.md`.

CMake exposes preparation switches for vendor libraries:

```bash
-DGW_ENABLE_BLAS_LAPACK=ON
-DGW_ENABLE_SCALAPACK=ON
-DGW_ENABLE_COSMA=ON
-DGW_ENABLE_CUDA=ON
```

These switches only prepare/link the relevant vendor targets when available. The actual optimized backend classes should be added as separate implementations of `gw::linalg::Backend`.
