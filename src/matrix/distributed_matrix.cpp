#include "gw/matrix/distributed_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef GW_ENABLE_MPI
#include <mpi.h>
#else
#error "distributed_matrix.cpp requires GW_ENABLE_MPI"
#endif

namespace gw::matrix {
namespace {

extern "C" {
int numroc_(const int* n, const int* nb, const int* iproc, const int* isrcproc, const int* nprocs);
void descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb,
               const int* irsrc, const int* icsrc, const int* ictxt, const int* lld, int* info);

void Cblacs_get(int context, int request, int* value);
int Csys2blacs_handle(MPI_Comm communicator);
void Cblacs_gridinit(int* context, const char* order, int nprow, int npcol);
void Cblacs_gridinfo(int context, int* nprow, int* npcol, int* myrow, int* mycol);
void Cblacs_gridexit(int context);

void pzgemm_(const char* transa, const char* transb,
             const int* m, const int* n, const int* k,
             const Complex* alpha,
             const Complex* a, const int* ia, const int* ja, const int* desca,
             const Complex* b, const int* ib, const int* jb, const int* descb,
             const Complex* beta,
             Complex* c, const int* ic, const int* jc, const int* descc);
void pzgetrf_(const int* m, const int* n,
              Complex* a, const int* ia, const int* ja, const int* desca,
              int* ipiv, int* info);

void pzgetrs_(const char* trans, const int* n, const int* nrhs,
              const Complex* a, const int* ia, const int* ja, const int* desca,
              const int* ipiv,
              Complex* b, const int* ib, const int* jb, const int* descb,
              int* info);
}

[[nodiscard]] int checked_int(std::size_t value, const char* what) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("ScaLAPACK integer overflow for ") + what);
    }
    return static_cast<int>(value);
}

[[nodiscard]] int owner_process(std::size_t global_index, int block, int nprocs) noexcept {
    return static_cast<int>((global_index / static_cast<std::size_t>(block)) % static_cast<std::size_t>(nprocs));
}

[[nodiscard]] std::size_t local_index(std::size_t global_index, int block, int nprocs) noexcept {
    const std::size_t block_id = global_index / static_cast<std::size_t>(block);
    const std::size_t local_block = block_id / static_cast<std::size_t>(nprocs);
    return local_block * static_cast<std::size_t>(block) + global_index % static_cast<std::size_t>(block);
}

[[nodiscard]] char trans_char(linalg::MatrixTranspose trans) {
    switch (trans) {
        case linalg::MatrixTranspose::NoTranspose: return 'N';
        case linalg::MatrixTranspose::Transpose: return 'T';
        case linalg::MatrixTranspose::ConjugateTranspose: return 'C';
    }
    throw std::runtime_error("unknown MatrixTranspose value");
}

[[nodiscard]] std::size_t effective_rows(const DistributedMatrixComplex& a, linalg::MatrixTranspose trans) noexcept {
    return trans == linalg::MatrixTranspose::NoTranspose ? a.global_rows() : a.global_cols();
}

[[nodiscard]] std::size_t effective_cols(const DistributedMatrixComplex& a, linalg::MatrixTranspose trans) noexcept {
    return trans == linalg::MatrixTranspose::NoTranspose ? a.global_cols() : a.global_rows();
}

} // namespace

BlacsGrid::BlacsGrid(int ranks_per_group) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        throw std::runtime_error("BlacsGrid requires initialized MPI");
    }

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int effective_group_size = ranks_per_group <= 0 ? world_size : std::min(ranks_per_group, world_size);
    effective_group_size = std::max(1, effective_group_size);
    const int color = world_rank / effective_group_size;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &communicator_);
    owns_communicator_ = true;

    MPI_Comm_rank(communicator_, &rank_);
    MPI_Comm_size(communicator_, &size_);

    nprow_ = static_cast<int>(std::floor(std::sqrt(static_cast<double>(size_))));
    while (nprow_ > 1 && size_ % nprow_ != 0) {
        --nprow_;
    }
    npcol_ = size_ / nprow_;

    context_ = Csys2blacs_handle(communicator_);
    Cblacs_gridinit(&context_, "Row-major", nprow_, npcol_);
    Cblacs_gridinfo(context_, &nprow_, &npcol_, &myrow_, &mycol_);
}

BlacsGrid::~BlacsGrid() {
    if (context_ >= 0) {
        Cblacs_gridexit(context_);
    }
    if (owns_communicator_ && communicator_ != MPI_COMM_NULL) {
        MPI_Comm_free(&communicator_);
        communicator_ = MPI_COMM_NULL;
    }
}

DistributedMatrixComplex::DistributedMatrixComplex(std::shared_ptr<const BlacsGrid> grid,
                                                   std::size_t global_rows,
                                                   std::size_t global_cols,
                                                   int block_rows,
                                                   int block_cols)
    : grid_(std::move(grid)),
      global_rows_(global_rows),
      global_cols_(global_cols),
      global_rows_int_(checked_int(global_rows, "global rows")),
      global_cols_int_(checked_int(global_cols, "global cols")),
      block_rows_(std::max(1, std::min(block_rows, std::max(1, global_rows_int_)))),
      block_cols_(std::max(1, std::min(block_cols, std::max(1, global_cols_int_)))) {
    if (!grid_) {
        throw std::runtime_error("DistributedMatrixComplex: null BLACS grid");
    }

    const int zero = 0;
    int myrow = grid_->myrow();
    int mycol = grid_->mycol();
    int nprow = grid_->nprow();
    int npcol = grid_->npcol();
    local_rows_ = numroc_(&global_rows_int_, &block_rows_, &myrow, &zero, &nprow);
    local_cols_ = numroc_(&global_cols_int_, &block_cols_, &mycol, &zero, &npcol);
    lld_ = std::max(1, local_rows_);
    local_.assign(static_cast<std::size_t>(lld_) * static_cast<std::size_t>(std::max(1, local_cols_)),
                  Complex{0.0, 0.0});

    int info = 0;
    int ictxt = grid_->context();
    descinit_(desc_.data(),
              &global_rows_int_,
              &global_cols_int_,
              &block_rows_,
              &block_cols_,
              &zero,
              &zero,
              &ictxt,
              &lld_,
              &info);
    if (info != 0) {
        throw std::runtime_error("descinit_ failed with info=" + std::to_string(info));
    }
}

bool DistributedMatrixComplex::owns_global(std::size_t global_i, std::size_t global_j) const noexcept {
    return owner_process(global_i, block_rows_, grid_->nprow()) == grid_->myrow() &&
           owner_process(global_j, block_cols_, grid_->npcol()) == grid_->mycol();
}

std::size_t DistributedMatrixComplex::local_row(std::size_t global_i) const noexcept {
    return local_index(global_i, block_rows_, grid_->nprow());
}

std::size_t DistributedMatrixComplex::local_col(std::size_t global_j) const noexcept {
    return local_index(global_j, block_cols_, grid_->npcol());
}

Complex& DistributedMatrixComplex::local_at(std::size_t local_i, std::size_t local_j) {
    return local_[local_i + local_j * static_cast<std::size_t>(lld_)];
}

const Complex& DistributedMatrixComplex::local_at(std::size_t local_i, std::size_t local_j) const {
    return local_[local_i + local_j * static_cast<std::size_t>(lld_)];
}

Complex& DistributedMatrixComplex::owned_global_at(std::size_t global_i, std::size_t global_j) {
    if (!owns_global(global_i, global_j)) {
        throw std::runtime_error("DistributedMatrixComplex::owned_global_at called for non-owned element");
    }
    return local_at(local_row(global_i), local_col(global_j));
}

const Complex& DistributedMatrixComplex::owned_global_at(std::size_t global_i, std::size_t global_j) const {
    if (!owns_global(global_i, global_j)) {
        throw std::runtime_error("DistributedMatrixComplex::owned_global_at called for non-owned element");
    }
    return local_at(local_row(global_i), local_col(global_j));
}

void DistributedMatrixComplex::fill(Complex value) {
    std::fill(local_.begin(), local_.end(), value);
}

void DistributedMatrixComplex::for_each_owned_global(const std::function<void(std::size_t, std::size_t, Complex&)>& fn) {
    for (std::size_t i = 0; i < global_rows_; ++i) {
        if (owner_process(i, block_rows_, grid_->nprow()) != grid_->myrow()) {
            continue;
        }
        const std::size_t li = local_row(i);
        for (std::size_t j = 0; j < global_cols_; ++j) {
            if (owner_process(j, block_cols_, grid_->npcol()) != grid_->mycol()) {
                continue;
            }
            const std::size_t lj = local_col(j);
            fn(i, j, local_at(li, lj));
        }
    }
}

void DistributedMatrixComplex::for_each_owned_global(const std::function<void(std::size_t, std::size_t, const Complex&)>& fn) const {
    for (std::size_t i = 0; i < global_rows_; ++i) {
        if (owner_process(i, block_rows_, grid_->nprow()) != grid_->myrow()) {
            continue;
        }
        const std::size_t li = local_row(i);
        for (std::size_t j = 0; j < global_cols_; ++j) {
            if (owner_process(j, block_cols_, grid_->npcol()) != grid_->mycol()) {
                continue;
            }
            const std::size_t lj = local_col(j);
            fn(i, j, local_at(li, lj));
        }
    }
}

std::shared_ptr<const BlacsGrid> make_blacs_grid(int ranks_per_group) {
    return std::make_shared<BlacsGrid>(ranks_per_group);
}

DistributedMatrixComplex distributed_identity(std::shared_ptr<const BlacsGrid> grid,
                                              std::size_t n,
                                              int block_size) {
    DistributedMatrixComplex out(std::move(grid), n, n, block_size, block_size);
    out.for_each_owned_global([](std::size_t i, std::size_t j, Complex& value) {
        value = (i == j) ? Complex{1.0, 0.0} : Complex{0.0, 0.0};
    });
    return out;
}

DistributedMatrixComplex scatter_replicated_to_distributed(const MatrixComplex& src,
                                                           std::shared_ptr<const BlacsGrid> grid,
                                                           int block_size) {
    DistributedMatrixComplex out(grid, src.rows(), src.cols(), block_size, block_size);
    out.for_each_owned_global([&](std::size_t i, std::size_t j, Complex& value) {
        value = src(i, j);
    });
    return out;
}

DistributedMatrixComplex scatter_real_panel_to_distributed(const MatrixReal& src,
                                                           std::shared_ptr<const BlacsGrid> grid,
                                                           int block_size) {
    DistributedMatrixComplex out(grid, src.rows(), src.cols(), block_size, block_size);
    out.for_each_owned_global([&](std::size_t i, std::size_t j, Complex& value) {
        value = Complex{src(i, j), 0.0};
    });
    return out;
}

MatrixComplex gather_distributed_to_replicated(const DistributedMatrixComplex& src) {
    MatrixComplex local_full(src.global_rows(), src.global_cols(), Complex{0.0, 0.0});
    src.for_each_owned_global([&](std::size_t i, std::size_t j, const Complex& value) {
        local_full(i, j) = value;
    });

    MatrixComplex global(local_full.rows(), local_full.cols(), Complex{0.0, 0.0});
    const auto count = checked_int(global.size() * 2, "MPI complex gather buffer length in doubles");
    MPI_Allreduce(local_full.data().data(),
                  global.data().data(),
                  count,
                  MPI_DOUBLE,
                  MPI_SUM,
                  src.grid()->communicator());
    return global;
}

DistributedMatrixComplex distributed_gemm(const DistributedMatrixComplex& a,
                                          const DistributedMatrixComplex& b,
                                          linalg::MatrixTranspose trans_a,
                                          linalg::MatrixTranspose trans_b) {
    if (a.grid().get() != b.grid().get()) {
        throw std::runtime_error("distributed_gemm: matrices must use the same BLACS grid object");
    }
    if (effective_cols(a, trans_a) != effective_rows(b, trans_b)) {
        throw std::runtime_error("distributed_gemm: incompatible matrix shapes");
    }

    auto grid = a.grid();
    DistributedMatrixComplex c(grid, effective_rows(a, trans_a), effective_cols(b, trans_b), a.block_rows(), b.block_cols());

    const char ta = trans_char(trans_a);
    const char tb = trans_char(trans_b);
    const int one = 1;
    const int m = c.global_rows_int();
    const int n = c.global_cols_int();
    const int k = checked_int(effective_cols(a, trans_a), "distributed GEMM k dimension");
    const Complex alpha{1.0, 0.0};
    const Complex beta{0.0, 0.0};

    pzgemm_(&ta,
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
}

DistributedMatrixComplex distributed_inverse_by_solve(const DistributedMatrixComplex& a) {
    if (a.global_rows() != a.global_cols()) {
        throw std::runtime_error("distributed_inverse_by_solve: matrix must be square");
    }
    DistributedMatrixComplex lu = a;
    DistributedMatrixComplex x = distributed_identity(a.grid(), a.global_rows(), a.block_rows());

    const int n = lu.global_rows_int();
    const int one = 1;
    const int nrhs = n;
    std::vector<int> ipiv(static_cast<std::size_t>(lu.local_rows()) + static_cast<std::size_t>(lu.block_rows()) + 1U);
    int info = 0;

    pzgetrf_(&n, &n, lu.local_data_ptr(), &one, &one, lu.descriptor(), ipiv.data(), &info);
    if (info != 0) {
        throw std::runtime_error("pzgetrf_ failed with info=" + std::to_string(info));
    }

    const char trans = 'N';
    pzgetrs_(&trans,
             &n,
             &nrhs,
             lu.local_data_ptr(),
             &one,
             &one,
             lu.descriptor(),
             ipiv.data(),
             x.local_data_ptr(),
             &one,
             &one,
             x.descriptor(),
             &info);
    if (info != 0) {
        throw std::runtime_error("pzgetrs_ failed with info=" + std::to_string(info));
    }

    return x;
}

std::vector<Complex> distributed_column_dot_same_layout(const DistributedMatrixComplex& x,
                                                               const DistributedMatrixComplex& y) {
    if (x.grid().get() != y.grid().get()) {
        throw std::runtime_error("distributed_column_dot_same_layout: matrices must use the same BLACS grid object");
    }
    if (x.global_rows() != y.global_rows() || x.global_cols() != y.global_cols()) {
        throw std::runtime_error("distributed_column_dot_same_layout: incompatible matrix shapes");
    }
    if (x.block_rows() != y.block_rows() || x.block_cols() != y.block_cols()) {
        throw std::runtime_error("distributed_column_dot_same_layout: incompatible block sizes");
    }

    std::vector<Complex> local_values(x.global_cols(), Complex{0.0, 0.0});
    y.for_each_owned_global([&](std::size_t row, std::size_t col, const Complex& y_value) {
        local_values[col] += x.owned_global_at(row, col) * y_value;
    });
    return mpi_allreduce_sum_complex_vector(local_values, *x.grid());
}

Complex mpi_allreduce_sum_complex(Complex value, const BlacsGrid& grid) {
    double send[2] = {value.real(), value.imag()};
    double recv[2] = {0.0, 0.0};
    MPI_Allreduce(send, recv, 2, MPI_DOUBLE, MPI_SUM, grid.communicator());
    return Complex{recv[0], recv[1]};
}


std::vector<Complex> mpi_allreduce_sum_complex_vector(const std::vector<Complex>& values, const BlacsGrid& grid) {
    std::vector<double> send(2U * values.size(), 0.0);
    std::vector<double> recv(2U * values.size(), 0.0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        send[2U * i] = values[i].real();
        send[2U * i + 1U] = values[i].imag();
    }
    MPI_Allreduce(send.data(), recv.data(), static_cast<int>(recv.size()), MPI_DOUBLE, MPI_SUM, grid.communicator());
    std::vector<Complex> out(values.size(), Complex{0.0, 0.0});
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = Complex{recv[2U * i], recv[2U * i + 1U]};
    }
    return out;
}

} // namespace gw::matrix
