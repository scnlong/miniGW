#pragma once

#include "gw/linalg.hpp"
#include "gw/types.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#ifdef GW_ENABLE_MPI
#include <mpi.h>
#endif

namespace gw::matrix {

enum class DistributedGemmProvider {
    ScalapackPzgemm,
    CosmaAdapter
};

// A 2-D BLACS block-cyclic complex matrix used by the ScaLAPACK path.
// The local storage is column-major because ScaLAPACK expects Fortran layout.
class BlacsGrid {
public:
    explicit BlacsGrid(int ranks_per_group = 0);
    ~BlacsGrid();

    BlacsGrid(const BlacsGrid&) = delete;
    BlacsGrid& operator=(const BlacsGrid&) = delete;

    [[nodiscard]] int context() const noexcept { return context_; }
    [[nodiscard]] int nprow() const noexcept { return nprow_; }
    [[nodiscard]] int npcol() const noexcept { return npcol_; }
    [[nodiscard]] int myrow() const noexcept { return myrow_; }
    [[nodiscard]] int mycol() const noexcept { return mycol_; }
    [[nodiscard]] int rank() const noexcept { return rank_; }
    [[nodiscard]] int size() const noexcept { return size_; }
#ifdef GW_ENABLE_MPI
    [[nodiscard]] MPI_Comm communicator() const noexcept { return communicator_; }
#endif

private:
    int context_{-1};
    int nprow_{1};
    int npcol_{1};
    int myrow_{0};
    int mycol_{0};
    int rank_{0};
    int size_{1};
#ifdef GW_ENABLE_MPI
    MPI_Comm communicator_{MPI_COMM_NULL};
    bool owns_communicator_{false};
#endif
};

class DistributedMatrixComplex {
public:
    static constexpr int descriptor_size = 9;

    DistributedMatrixComplex() = default;
    DistributedMatrixComplex(std::shared_ptr<const BlacsGrid> grid,
                             std::size_t global_rows,
                             std::size_t global_cols,
                             int block_rows = 64,
                             int block_cols = 64);

    [[nodiscard]] const std::shared_ptr<const BlacsGrid>& grid() const noexcept { return grid_; }

    [[nodiscard]] std::size_t global_rows() const noexcept { return global_rows_; }
    [[nodiscard]] std::size_t global_cols() const noexcept { return global_cols_; }
    [[nodiscard]] int global_rows_int() const noexcept { return global_rows_int_; }
    [[nodiscard]] int global_cols_int() const noexcept { return global_cols_int_; }

    [[nodiscard]] int block_rows() const noexcept { return block_rows_; }
    [[nodiscard]] int block_cols() const noexcept { return block_cols_; }
    [[nodiscard]] int local_rows() const noexcept { return local_rows_; }
    [[nodiscard]] int local_cols() const noexcept { return local_cols_; }
    [[nodiscard]] int lld() const noexcept { return lld_; }

    [[nodiscard]] const int* descriptor() const noexcept { return desc_.data(); }
    [[nodiscard]] int* descriptor() noexcept { return desc_.data(); }

    [[nodiscard]] const std::vector<Complex>& local_data() const noexcept { return local_; }
    [[nodiscard]] std::vector<Complex>& local_data() noexcept { return local_; }
    [[nodiscard]] const Complex* local_data_ptr() const noexcept { return local_.data(); }
    [[nodiscard]] Complex* local_data_ptr() noexcept { return local_.data(); }

    [[nodiscard]] bool owns_global(std::size_t global_i, std::size_t global_j) const noexcept;
    [[nodiscard]] std::size_t local_row(std::size_t global_i) const noexcept;
    [[nodiscard]] std::size_t local_col(std::size_t global_j) const noexcept;

    [[nodiscard]] Complex& local_at(std::size_t local_i, std::size_t local_j);
    [[nodiscard]] const Complex& local_at(std::size_t local_i, std::size_t local_j) const;

    [[nodiscard]] Complex& owned_global_at(std::size_t global_i, std::size_t global_j);
    [[nodiscard]] const Complex& owned_global_at(std::size_t global_i, std::size_t global_j) const;

    void fill(Complex value);

    // Iterate over the global indices owned by this MPI rank.
    void for_each_owned_global(const std::function<void(std::size_t, std::size_t, Complex&)>& fn);
    void for_each_owned_global(const std::function<void(std::size_t, std::size_t, const Complex&)>& fn) const;

private:
    std::shared_ptr<const BlacsGrid> grid_{};
    std::size_t global_rows_{0};
    std::size_t global_cols_{0};
    int global_rows_int_{0};
    int global_cols_int_{0};
    int block_rows_{64};
    int block_cols_{64};
    int local_rows_{0};
    int local_cols_{0};
    int lld_{1};
    std::array<int, descriptor_size> desc_{};
    std::vector<Complex> local_{};
};

[[nodiscard]] std::shared_ptr<const BlacsGrid> make_blacs_grid(int ranks_per_group = 0);

[[nodiscard]] DistributedMatrixComplex distributed_identity(std::shared_ptr<const BlacsGrid> grid,
                                                           std::size_t n,
                                                           int block_size = 64);

[[nodiscard]] DistributedMatrixComplex scatter_replicated_to_distributed(const MatrixComplex& src,
                                                                         std::shared_ptr<const BlacsGrid> grid,
                                                                         int block_size = 64);

[[nodiscard]] DistributedMatrixComplex scatter_real_panel_to_distributed(const MatrixReal& src,
                                                                        std::shared_ptr<const BlacsGrid> grid,
                                                                        int block_size = 64);

[[nodiscard]] MatrixComplex gather_distributed_to_replicated(const DistributedMatrixComplex& src);

[[nodiscard]] DistributedMatrixComplex distributed_gemm(const DistributedMatrixComplex& a,
                                                       const DistributedMatrixComplex& b,
                                                       linalg::MatrixTranspose trans_a = linalg::MatrixTranspose::NoTranspose,
                                                       linalg::MatrixTranspose trans_b = linalg::MatrixTranspose::NoTranspose,
                                                       DistributedGemmProvider provider = DistributedGemmProvider::ScalapackPzgemm);

[[nodiscard]] DistributedMatrixComplex distributed_inverse_by_solve(const DistributedMatrixComplex& a);

[[nodiscard]] Complex mpi_allreduce_sum_complex(Complex value, const BlacsGrid& grid);
[[nodiscard]] std::vector<Complex> distributed_column_dot_same_layout(const DistributedMatrixComplex& x,
                                                                      const DistributedMatrixComplex& y);

[[nodiscard]] std::vector<Complex> mpi_allreduce_sum_complex_vector(const std::vector<Complex>& values, const BlacsGrid& grid);

} // namespace gw::matrix
