#pragma once

#include "matrix/distributed_matrix.hpp"

namespace gw::matrix {

// Explicit COSMA distributed GEMM wrapper.  This function calls COSMA's
// prefixed PBLAS ABI symbol cosma_pzgemm_.  It is deliberately kept out of
// distributed_matrix.cpp so ordinary ScaLAPACK code remains free of COSMA
// symbols and dependencies.
[[nodiscard]] DistributedMatrixComplex cosma_distributed_gemm(
    const DistributedMatrixComplex& a,
    const DistributedMatrixComplex& b,
    linalg::MatrixTranspose trans_a = linalg::MatrixTranspose::NoTranspose,
    linalg::MatrixTranspose trans_b = linalg::MatrixTranspose::NoTranspose);

} // namespace gw::matrix
