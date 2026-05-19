# miniGW formulation notes

This document summarizes the equations and index conventions implemented in the current miniGW prototype.  It is intentionally close to the C++ implementation rather than a complete derivation of the GW approximation.

The code uses 0-based indexing throughout.

## Orbital and particle-hole indices

Molecular orbitals are split into occupied and virtual spaces:

```text
i, j, k_occ   occupied MO indices: 0 ... nocc-1
a, b          virtual MO indices:  nocc ... nmo-1
p, q, k       general MO indices:  0 ... nmo-1
```

The local virtual index is:

```text
a_vir = a_mo - nocc
```

The particle-hole index is:

```text
ph = i_occ * nvir + a_vir
```

with inverse mapping:

```text
i_occ = ph / nvir
a_vir = ph % nvir
a_mo  = nocc + a_vir
```

For a particle-hole pair `ia`, the stored excitation energy is:

```text
Delta_ia = epsilon_a - epsilon_i
```

## Input quantities

The PySCF HDF5 input bundle provides:

```text
mo_energy[p]
eri_mo[p,q,r,s]
vxc_mo[p,q]
nocc
fermi_energy
```

The ERI accessor in the C++ code preserves the exact HDF5 index order:

```cpp
integrals.eri(p, q, r, s) == eri_mo[p,q,r,s]
```

The current implementation assumes this index convention consistently in the exchange, particle-hole matrix, and contraction-panel builders.

## Imaginary-frequency grid

miniGW builds an imaginary-frequency integration grid through `gw::transformed_gauss_legendre_grid(...)`.  The grid stores frequencies and quadrature weights:

```text
omega_im[n] = i * omega_n
weights[n]
```

The correlation self-energy is evaluated on this imaginary-frequency grid before Padé analytic continuation.

## Exchange self-energy

The exchange matrix is computed as:

```text
Sigma_x[p,q] = - sum_i^occ eri_mo[p,i,q,i]
```

This is the dense MO-basis exchange contribution used later in the diagonal quasiparticle equation.

## Particle-hole Coulomb matrix

For particle-hole indices `ia` and `jb`, the current convention is:

```text
V_ph[jb, ia] = eri_mo[j,b,i,a]
```

Equivalently, `row = jb` and `col = ia`.  This convention is important because the dielectric matrix is built as:

```text
epsilon[row, col] = delta[row, col] - V_ph[row, col] * Pi0[col]
```

where `Pi0` is diagonal in the particle-hole basis.

## Diagonal independent-particle polarizability

For a particle-hole pair `ia` with `Delta_ia = epsilon_a - epsilon_i`, the implemented diagonal polarizability is:

```text
Pi0_ia(i omega) = 2 * [ 1 / (i omega - Delta_ia + i eta)
                       - 1 / (i omega + Delta_ia - i eta) ]
```

The factor of two corresponds to the spin-degenerate closed-shell convention used by this prototype.

## Dielectric matrix and screened interaction

For each integration frequency `i omega'`, miniGW forms the diagonal independent-particle polarizability `D = diag(Pi0)` and uses the left-dielectric representation

```text
epsilon_left = I - D V_ph
```

that is,

```text
epsilon_left[row, col] = delta[row, col] - Pi0[row] * V_ph[row, col].
```

The host, ScaLAPACK, and COSMA screening workspaces then compute

```text
W_c = epsilon_left^{-1} D
    = (I - diag(Pi0) V_ph)^(-1) diag(Pi0).
```

This replaces the older algebraically equivalent form

```text
W_c = [(I - V_ph diag(Pi0))^{-1} - I]^T V_ph^{-1}
```

and avoids explicit construction of `V_ph^{-1}`, which is numerically unsafe in large particle-hole product spaces.

## Panel contraction for the diagonal correlation self-energy

For a selected external state `p`, the code loops over general MO indices `k` in panels.  A panel contains:

```text
X(ph, kk) = eri_mo[p, k_begin + kk, i, a]
```

where `ph -> (i,a)` and `kk = 0 ... panel_width-1`.

The screened-interaction contraction for each column is evaluated as:

```text
Y = W_c X
q_kk = X[:,kk]^T Y[:,kk]
```

The local path performs this with dense host GEMM.  The ScaLAPACK path performs the same multiplication with distributed PBLAS/ScaLAPACK.  The COSMA path routes the distributed GEMM operations to the COSMA `cosma_pzgemm_` provider.

For an external imaginary frequency `i omega_n`, the implemented accumulation is:

```text
Sigma_c[p, i omega_n] = -1/pi * sum_{omega'} sum_k
    G0_factor(k, omega_n, omega') * q(p,k,omega') * weight(omega')
```

with:

```text
G0_factor = (i omega_n + fermi_energy - epsilon_k)
            / [ (i omega_n + fermi_energy - epsilon_k)^2 - (i omega')^2 ]
```

This is written in the code as an accumulation over `f_prime`, `p`, and `k` panels.

## Padé continuation and quasiparticle update

For each requested diagonal state, miniGW fits the imaginary-axis correlation self-energy with a continued-fraction Padé approximation.  The diagonal quasiparticle equation is then solved iteratively as:

```text
E_QP[p] = epsilon_p
          + Re Sigma_c[p](E_QP[p] - fermi_energy)
          + Sigma_x[p,p]
          - vxc_mo[p,p]
```

The implementation currently uses only the diagonal quasiparticle update; off-diagonal quasiparticle Hamiltonian diagonalization is not implemented.

## Parallel execution paths

The formulation above is shared by all execution paths.  The implementation differs in data ownership and linear-algebra provider:

| Path | Screening matrix ownership | GEMM / inversion provider |
|---|---|---|
| Reference | replicated host | internal reference backend |
| BLAS/LAPACK | replicated host | Fortran BLAS/LAPACK |
| MPI frequency distribution | replicated per MPI rank | local backend per assigned frequency |
| ScaLAPACK | BLACS block-cyclic distributed | ScaLAPACK/PBLAS |
| COSMA | distributed screening matrices | COSMA `cosma_pzgemm_` for GEMM; ScaLAPACK for remaining distributed solve/factorization infrastructure |
| CUDA cuBLAS/cuSolver | per-rank device-resident dense screening workspace | cuBLAS/cuSolver |

## Scientific and scaling limitations

- RI / density fitting is not implemented.
- The input is a full four-index MO ERI tensor.
- The ERI tensor is still replicated in host memory.
- The code computes diagonal quasiparticle updates only.
- The ScaLAPACK path distributes the dominant CPU screening matrices but does not distribute the four-index ERI input.
- The COSMA path targets distributed GEMM in the screening workflow; it is not a complete ScaLAPACK replacement and does not solve integral storage.
- CUDA/cuBLAS support is per-rank and device-oriented; it still starts from replicated host input data and does not distribute one screening matrix across multiple GPUs.
