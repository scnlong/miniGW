# gw_prototype

A C++20 G0W0 prototype.

This project intentionally starts with a serial CPU implementation and a small self-contained `.npy` reader. MPI, CUDA, and GoogleTest can be added later behind CMake options.

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
- minimal NumPy `.npy` reader for little-endian `float64` arrays.

## Expected input files

Run from a directory containing:

```text
eri_mo.npy
mo_energy.npy
vxc_mo.npy
nocc.txt
fermi_energy.txt
```

The `.npy` reader currently supports C-order, little-endian `float64` arrays only.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

```bash
./build/gw20 --input-dir /path/to/pyscf_output --freq-points 200 --pade-params 16 --state 5
```

Use `--all-states` to compute all diagonal states. The default follows the Julia prototype and computes state 5.
