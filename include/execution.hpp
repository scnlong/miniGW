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
    std::size_t mpi_local_rank{0};
    std::size_t mpi_local_size{1};
    std::size_t frequency_workspace_replicas{1};

    // ScaLAPACK/COSMA can use two levels of MPI parallelism:
    //   - frequency groups: independent communicator groups process different frequency points;
    //   - ranks within each group: collectively execute BLACS/ScaLAPACK operations.
    // For non-distributed backends these fields are informational and unused.
    std::size_t scalapack_ranks_per_group{4};
    std::size_t frequency_group_id{0};
    std::size_t num_frequency_groups{1};
    std::size_t frequency_group_rank{0};
    std::size_t frequency_group_size{1};

    // GPU device sharing policy for replicated MPI frequency parallelism.
    // Device id is chosen as (mpi_local_rank / tasks_per_gpu) % visible_device_count.
    std::size_t tasks_per_gpu{4};
    int cuda_device_id{-1};
};

[[nodiscard]] std::string_view to_string(FrequencyParallelMode mode) noexcept;
[[nodiscard]] FrequencyParallelMode choose_frequency_parallel_mode(const Cli& cli, const MpiContext& mpi);
[[nodiscard]] std::size_t frequency_workspace_replicas(FrequencyParallelMode mode);
[[nodiscard]] bool choose_openmp_kernel_loops(const Cli& cli, FrequencyParallelMode frequency_mode);
void configure_scalapack_frequency_groups(ExecutionPolicy& policy, const Cli& cli);

} // namespace gw
