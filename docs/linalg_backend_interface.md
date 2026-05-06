# Linear-algebra backend boundary

This version keeps the default executable self-contained, but it removes the hard dependency of the GW equations on hand-written dense linear algebra.

The interface is `gw::linalg::Backend` in `include/gw/linalg.hpp`. The default implementation is `gw::linalg::ReferenceBackend` in `src/linalg.cpp`.

Current backend operations are deliberately minimal:

- `inverse(MatrixComplex)` for the complex dielectric matrix and Coulomb matrix inversions;
- `gemm(A, B, trans_a, trans_b)` for matrix-matrix products such as `inv_eps^T * inv_v`;
- `gemv(A, x, trans_a)` for complex matrix times real vector;
- `quadratic_form(x, A)` for contractions of the form `x^T A x`.

These operations are the immediate replacement points for external libraries:

- OpenBLAS/LAPACK: replace `inverse` with LU factorization plus solve/inversion, and replace `gemm`/`gemv` with BLAS calls. Because the project stores matrices in row-major order, the wrapper must either translate transpose flags carefully or introduce a column-major matrix view.
- ScaLAPACK: replace `inverse` and large matrix products after introducing a distributed matrix descriptor. This requires a new distributed matrix type or a view object; the current `MatrixComplex` is a single-process owner.
- COSMA: use it for distributed dense matrix multiplication after the particle-hole matrices are distributed. COSMA is a GEMM backend, not a replacement for dielectric inversion.
- cuBLAS/cuSolver: use cuBLAS for `gemm`/`gemv` and cuSolver for LU-based complex inversions. This requires an explicit device-memory owner and transfer policy; do not hide host-device copies inside every small backend call.

The important design point is that `src/gw.cpp` now calls the backend through `GwSettings::linalg_backend`; the physics-level code no longer directly contains the matrix inversion or dense contraction kernels.
