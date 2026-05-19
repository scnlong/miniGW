# Data layout and index semantics

This document records the index conventions and in-memory data layout used by miniGW.  Its main purpose is to make the orbital, particle-hole, ERI, and panel-contraction indices explicit enough that backend implementations can be checked against the same semantics.

The C++ implementation uses 0-based indexing throughout.

## Input bundle

miniGW reads one PySCF-generated HDF5 input bundle from the input directory:

```text
gw_input.h5
```

The file is read by `src/hdf5_input.cpp` through the HDF5 C API.  The required datasets are:

```text
/mo_energy   shape: (nmo,)                 float64
/eri_mo      shape: (nmo, nmo, nmo, nmo)   float64
/vxc_mo      shape: (nmo, nmo)             float64
```

The required file attributes are:

```text
nocc          integer
fermi_energy  float64
```

All energies are in Hartree.  The arrays are stored in C-order and are copied directly into miniGW's row-major containers.

## Array containers

The project uses lightweight row-major containers in `include/types.hpp`:

```cpp
namespace gw {

template <typename T> class Matrix;
template <typename T> class Tensor3;
template <typename T> class Tensor4;

using MatrixReal    = Matrix<double>;
using MatrixComplex = Matrix<std::complex<double>>;
using Tensor3Real   = Tensor3<double>;
using Tensor4Real   = Tensor4<double>;

} // namespace gw
```

For a rank-2 matrix `A(i,j)` with dimensions `(n0,n1)`, the offset is:

```text
i * n1 + j
```

For a rank-3 tensor `T(i,j,k)` with dimensions `(n0,n1,n2)`, the offset is:

```text
(i * n1 + j) * n2 + k
```

For a rank-4 tensor `T(i,j,k,l)` with dimensions `(n0,n1,n2,n3)`, the offset is:

```text
(((i * n1 + j) * n2 + k) * n3 + l)
```

This layout matches the HDF5 payload written by the PySCF preparation script.

## Orbital space

`gw::OrbitalSpace` in `include/gw/orbital_space.hpp` owns the MO energies, the number of occupied orbitals, and the Fermi energy.  It exposes explicit accessors for common orbital-index operations:

```cpp
nmo()
nocc()
nvir()
fermi_energy()
energy(p)
is_occupied(p)
is_virtual(p)
virtual_to_mo(a_vir)
mo_to_virtual(a_mo)
```

The split between occupied and virtual orbitals is:

```text
occupied MO indices:  i = 0, ..., nocc - 1
virtual  MO indices:  a = nocc, ..., nmo - 1
local virtual index:  a_vir = a_mo - nocc
nvir = nmo - nocc
```

This avoids mixing a local virtual index with a global molecular-orbital index.

## Particle-hole basis

`gw::ParticleHoleBasis` in `include/gw/particle_hole.hpp` stores all occupied-virtual pairs as `gw::ParticleHolePair` objects:

```cpp
struct ParticleHolePair {
    std::size_t i_occ;  // global occupied MO index
    std::size_t a_vir;  // local virtual index, 0 ... nvir-1
    std::size_t a_mo;   // global virtual MO index
    double delta_e;     // epsilon_a - epsilon_i
};
```

The linear particle-hole index is:

```text
ph = i_occ * nvir + a_vir
```

and the inverse mapping is:

```text
i_occ = ph / nvir
a_vir = ph % nvir
a_mo  = nocc + a_vir
```

The stored excitation energy is:

```text
delta_e(ph) = epsilon[a_mo] - epsilon[i_occ]
```

## Molecular integrals

`gw::MolecularIntegrals` in `include/gw/integrals.hpp` owns the four-index MO ERI tensor and the `vxc_mo` matrix.  All accesses go through:

```cpp
integrals.eri(p, q, r, s)
integrals.vxc(p, q)
```

`integrals.eri(p,q,r,s)` returns the exact C-order payload read from `/eri_mo`.  miniGW does not currently transform or permute the tensor during input.

The current code uses the following ERI access patterns:

```text
Sigma_x[p,q]     = - sum_i^occ eri(p, i, q, i)
V_ph[jb, ia]     = eri(j, b, i, a)
P_panel[ia, kk]  = eri(p, k_begin + kk, i, a)
```

where `ia` and `jb` are particle-hole indices and `(i,a)` / `(j,b)` are the corresponding occupied-virtual pairs.

## Screening matrices

The dominant screening matrices live in the particle-hole basis and have shape:

```text
nph x nph,  where nph = nocc * nvir
```

The local host path stores these matrices as replicated `MatrixComplex` objects.  The ScaLAPACK and COSMA distributed paths store the dominant screening matrices as BLACS block-cyclic `DistributedMatrixComplex` objects.

The historical right-dielectric form used column scaling by the diagonal independent-particle polarizability:

```text
epsilon_right[row, col] = delta[row, col] - V_ph[row, col] * Pi0[col]
```

The current host, CUDA device-resident, ScaLAPACK, and COSMA screening workspaces avoid explicitly forming `V_ph^{-1}`. They instead build the left-dielectric form

```text
epsilon_left[row, col] = delta[row, col] - Pi0[row] * V_ph[row, col]
```

and compute

```text
W_c = epsilon_left^{-1} * diag(Pi0)
    = (I - diag(Pi0) V_ph)^(-1) diag(Pi0).
```

For nonsingular symmetric `V_ph`, this is algebraically equivalent to the old expression `(epsilon_right^{-1} - I)^T * V_ph^{-1}`, but is numerically safer because it never constructs `V_ph^{-1}`.

## Panel-based `pq_ph` contraction

The old fully materialized tensor would have had shape:

```text
pq_ph(nmo, nmo, nph)
```

with memory scaling:

```text
O(nmo^2 * nph)
```

The current code avoids this resident allocation by using `gw::workspace::PqPhPanelView` in `include/workspace/pq_ph_panel.hpp`.  For fixed `p` and a panel of MO indices `k_begin ... k_begin + width - 1`, it builds:

```text
X(ph, kk) = eri(p, k_begin + kk, i, a)
```

where `ph -> (i,a)`.  Thus:

```text
X.shape = nph x width
```

The local path computes:

```text
Y = W_c X
q_kk = X[:,kk]^T Y[:,kk]
```

The ScaLAPACK/COSMA distributed paths scatter the same panel into a distributed matrix, apply the distributed `W_c`, and reduce only the resulting scalar quadratic forms.

This removes the full resident `pq_ph(nmo,nmo,nph)` tensor, but it does not yet solve the deeper integral-storage problem: the four-index `eri_mo(nmo,nmo,nmo,nmo)` input is still replicated in host memory.

## Current ownership summary

| Object | Current ownership | Notes |
|---|---|---|
| `mo_energy` | replicated host | Small vector. |
| `vxc_mo` | replicated host | Small/medium dense matrix. |
| `eri_mo` | replicated host | Main remaining scaling limitation. |
| `V_ph`, left dielectric matrix, `W_c` in local path | replicated host | Uses `MatrixReal` / `MatrixComplex`. |
| `V_ph`, left dielectric matrix, `W_c` in ScaLAPACK path | BLACS block-cyclic distributed | CPU distributed path. |
| `V_ph`, left dielectric matrix, `W_c` in COSMA path | BLACS/COSMA distributed | Intended multi-node/multi-GPU distributed GEMM provider. |
| `V_ph`, left dielectric matrix, `W_c` in CUDA device-resident path | GPU device memory | Uses `DeviceScreeningWorkspace`; no device `inv(V_ph)` is stored. |
| CUDA device screening workspace | per-rank device-resident dense matrices | Still starts from replicated host input; does not distribute one screening matrix across multiple GPUs. |
| `pq_ph` | panel-generated | Full tensor is not materialized in the main workflow. |

## Known limitations

- The current input requires a full four-index MO ERI tensor.
- RI / density fitting is not implemented.
- Parallel HDF5 input and distributed/tiled ERI ownership are not implemented.
- The panel contraction removes a large intermediate tensor but does not remove replicated ERI storage.
- ScaLAPACK distributes the dominant CPU screening matrices, not the full GW data model.
- COSMA targets distributed GEMM in the screening workflow, but does not remove replicated ERI ownership or replace the remaining distributed solve/factorization infrastructure.
- The cuBLAS path is per-rank GPU-resident and uses MPI for frequency distribution, not for distributed GPU matrix ownership.
