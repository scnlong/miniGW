# Data layout and index semantics

This version makes the most important GW indices explicit rather than passing raw `std::vector<double>` and hand-written offset formulas through the code.

## Array containers

The project uses light-weight row-major containers in `include/gw/types.hpp`:

- `gw::Matrix<T>` for rank-2 arrays.
- `gw::Tensor3<T>` for rank-3 arrays.
- `gw::Tensor4<T>` for rank-4 arrays.

The aliases used in the code are:

- `MatrixReal = Matrix<double>`
- `MatrixComplex = Matrix<std::complex<double>>`
- `Tensor3Real = Tensor3<double>`
- `Tensor4Real = Tensor4<double>`

The layout is C-order / row-major. For a rank-4 tensor, the offset is

```text
(((i * n1 + j) * n2 + k) * n3 + l)
```

The `.npy` reader currently accepts only C-order little-endian `float64` arrays.

## Orbital space

`gw::OrbitalSpace` stores the MO energies, the number of occupied orbitals, and the Fermi energy. It provides explicit methods for the most common orbital-index operations:

- `nmo()`
- `nocc()`
- `nvir()`
- `energy(p)`
- `virtual_to_mo(a_vir)`
- `mo_to_virtual(a_mo)`

This avoids mixing a local virtual index with a global molecular-orbital index.

## Particle-hole basis

`gw::ParticleHoleBasis` stores all occupied-virtual pairs as `ParticleHolePair` objects:

```cpp
struct ParticleHolePair {
    std::size_t i_occ;  // global occupied MO index
    std::size_t a_vir;  // local virtual index
    std::size_t a_mo;   // global virtual MO index
    double delta_e;     // epsilon_a - epsilon_i
};
```

The linear particle-hole index is still equivalent to the original convention,

```text
ph = i_occ * nvir + a_vir
```

but this formula is now localized inside `ParticleHoleBasis`.

## Molecular integrals

`gw::MolecularIntegrals` owns the four-index ERI tensor and the `vxc_mo` matrix. All ERI access now goes through

```cpp
integrals.eri(p, q, r, s)
integrals.vxc(p, q)
```

The ERI accessor preserves the exact index order read from `eri_mo.npy`, matching the original Julia/NPZ implementation. The notation must still be verified against reference fixtures before using this as a production scientific implementation.
