#include "gw/matrix/cosma_distributed_matrix.hpp"

#include <stdexcept>
#include <limits>
#include <string>

namespace gw::matrix {
namespace {

extern "C" {
void cosma_pzgemm_(const char* transa, const char* transb,
                   const int* m, const int* n, const int* k,
                   const Complex* alpha,
                   const Complex* a, const int* ia, const int* ja, const int* desca,
                   const Complex* b, const int* ib, const int* jb, const int* descb,
                   const Complex* beta,
                   Complex* c, const int* ic, const int* jc, const int* descc);
}

[[nodiscard]] char trans_char(linalg::MatrixTranspose trans) noexcept {
    return trans == linalg::MatrixTranspose::NoTranspose ? 'N' : 'T';
}

[[nodiscard]] std::size_t effective_rows(const DistributedMatrixComplex& a,
                                         linalg::MatrixTranspose trans) noexcept {
    return trans == linalg::MatrixTranspose::NoTranspose ? a.global_rows() : a.global_cols();
}

[[nodiscard]] std::size_t effective_cols(const DistributedMatrixComplex& a,
                                         linalg::MatrixTranspose trans) noexcept {
    return trans == linalg::MatrixTranspose::NoTranspose ? a.global_cols() : a.global_rows();
}

[[nodiscard]] int checked_int(std::size_t value, const char* label) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(label) + " exceeds ScaLAPACK integer range");
    }
    return static_cast<int>(value);
}

} // namespace

DistributedMatrixComplex cosma_distributed_gemm(const DistributedMatrixComplex& a,
                                                const DistributedMatrixComplex& b,
                                                linalg::MatrixTranspose trans_a,
                                                linalg::MatrixTranspose trans_b) {
#ifndef GW_USE_COSMA_PXGEMM
    (void)a;
    (void)b;
    (void)trans_a;
    (void)trans_b;
    throw std::runtime_error("cosma_distributed_gemm: miniGW was built without GW_USE_COSMA_PXGEMM");
#else
    if (a.grid().get() != b.grid().get()) {
        throw std::runtime_error("cosma_distributed_gemm: matrices must use the same BLACS grid object");
    }
    if (effective_cols(a, trans_a) != effective_rows(b, trans_b)) {
        throw std::runtime_error("cosma_distributed_gemm: incompatible matrix shapes");
    }

    auto grid = a.grid();
    DistributedMatrixComplex c(grid, effective_rows(a, trans_a), effective_cols(b, trans_b),
                               a.block_rows(), b.block_cols());

    const char ta = trans_char(trans_a);
    const char tb = trans_char(trans_b);
    const int one = 1;
    const int m = c.global_rows_int();
    const int n = c.global_cols_int();
    const int k = checked_int(effective_cols(a, trans_a), "COSMA distributed GEMM k dimension");
    const Complex alpha{1.0, 0.0};
    const Complex beta{0.0, 0.0};

    cosma_pzgemm_(&ta,
                  &tb,
                  &m,
                  &n,
                  &k,
                  &alpha,
                  a.local_data_ptr(),
                  &one,
                  &one,
                  a.descriptor(),
                  b.local_data_ptr(),
                  &one,
                  &one,
                  b.descriptor(),
                  &beta,
                  c.local_data_ptr(),
                  &one,
                  &one,
                  c.descriptor());
    return c;
#endif
}

} // namespace gw::matrix
