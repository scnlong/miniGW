#pragma once

#include "types.hpp"

#include <vector>

namespace gw {

class MpiContext {
public:
    MpiContext(int& argc, char**& argv);
    ~MpiContext();

    MpiContext(const MpiContext&) = delete;
    MpiContext& operator=(const MpiContext&) = delete;

    [[nodiscard]] int rank() const noexcept { return rank_; }
    [[nodiscard]] int size() const noexcept { return size_; }
    [[nodiscard]] int local_rank() const noexcept { return local_rank_; }
    [[nodiscard]] int local_size() const noexcept { return local_size_; }
    [[nodiscard]] bool root() const noexcept { return rank_ == 0; }
    [[nodiscard]] bool enabled() const noexcept { return size_ > 1; }

private:
    bool owns_mpi_{false};
    int rank_{0};
    int size_{1};
    int local_rank_{0};
    int local_size_{1};
};

void mpi_allreduce_sum_in_place(std::vector<Complex>& values);
void mpi_barrier();

} // namespace gw
