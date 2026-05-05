## Index conventions

The C++ implementation uses 0-based indexing.

Particle-hole mapping:

```text
idx = i_occ * nvirt + (a_mo - nocc)
i_occ = idx / nvirt
a_mo = nocc + idx % nvirt
```

where `i_occ` indexes occupied orbitals and `a_mo` indexes virtual orbitals in the full MO index space.

## Main computational path

1. Load `eri_mo`, `mo_energy`, `vxc_mo`, `nocc`, and `fermi_energy`.
2. Build the transformed Gauss-Legendre imaginary-frequency grid.
3. Compute the exchange matrix:

```text
Sigma_x[p,q] = - sum_k^occ eri_mo[p,k,q,k]
```

4. Build the particle-hole Coulomb matrix:

```text
V_ph[jb,ia] = eri_mo[j,b,i,a]
```

5. For each imaginary integration frequency, compute the diagonal `Pi0_ph` and the correlation part of the screened interaction in the particle-hole representation.
6. Contract `(p,k|ph) W_c(ph,ph') (ph'|p,k)` for the diagonal self-energy.
7. Fit the imaginary-axis self-energy by continued-fraction Pade approximation and solve the diagonal quasiparticle equation iteratively.
