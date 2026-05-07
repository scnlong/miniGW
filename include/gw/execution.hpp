#pragma once

#include <cstddef>
#include <string_view>

namespace gw {

struct Cli;
class MpiContext;

enum class FrequencyParallelMode {
    Serial,
    OpenMP,
    MPI
};

struct ExecutionPolicy {
    FrequencyParallelMode frequency_parallel_mode{FrequencyParallelMode::Serial};
    bool openmp_kernel_loops{false};
    bool openmp_frequency_parallel{false};
    std::size_t mpi_rank{0};
    std::size_t mpi_size{1};
    std::size_t frequency_workspace_replicas{1};
};

[[nodiscard]] std::string_view to_string(FrequencyParallelMode mode) noexcept;
[[nodiscard]] FrequencyParallelMode choose_frequency_parallel_mode(const Cli& cli, const MpiContext& mpi);
[[nodiscard]] std::size_t frequency_workspace_replicas(FrequencyParallelMode mode);
[[nodiscard]] bool choose_openmp_kernel_loops(const Cli& cli, FrequencyParallelMode frequency_mode);

} // namespace gw
