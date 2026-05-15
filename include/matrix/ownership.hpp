#pragma once

#include "types.hpp"

#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace gw::matrix {

enum class MemorySpace {
    Host,
    Device,
    DistributedHost
};

enum class Distribution {
    Replicated,
    BlockCyclic2D,
    DeviceResident
};

[[nodiscard]] constexpr std::string_view to_string(MemorySpace space) noexcept {
    switch (space) {
        case MemorySpace::Host: return "host";
        case MemorySpace::Device: return "device";
        case MemorySpace::DistributedHost: return "distributed-host";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view to_string(Distribution distribution) noexcept {
    switch (distribution) {
        case Distribution::Replicated: return "replicated";
        case Distribution::BlockCyclic2D: return "block-cyclic-2d";
        case Distribution::DeviceResident: return "device-resident";
    }
    return "unknown";
}

template <class T>
class HostMatrixView {
public:
    HostMatrixView() = default;
    HostMatrixView(T* data, std::size_t rows, std::size_t cols, std::size_t leading_dimension)
        : data_(data), rows_(rows), cols_(cols), leading_dimension_(leading_dimension) {
        if (rows_ > 0 && cols_ > 0 && data_ == nullptr) {
            throw std::runtime_error("HostMatrixView: non-empty view has null data");
        }
        if (cols_ > leading_dimension_) {
            throw std::runtime_error("HostMatrixView: leading dimension is smaller than number of columns");
        }
    }

    [[nodiscard]] T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t leading_dimension() const noexcept { return leading_dimension_; }

    [[nodiscard]] T& operator()(std::size_t i, std::size_t j) const {
        return data_[i * leading_dimension_ + j];
    }

private:
    T* data_{nullptr};
    std::size_t rows_{0};
    std::size_t cols_{0};
    std::size_t leading_dimension_{0};
};

struct DataOwnership {
    MemorySpace memory_space{MemorySpace::Host};
    Distribution distribution{Distribution::Replicated};
    std::size_t owner_rank{0};
    std::size_t mpi_size{1};
    int device_id{-1};
};

[[nodiscard]] inline DataOwnership replicated_host_ownership() noexcept {
    return DataOwnership{};
}

} // namespace gw::matrix
