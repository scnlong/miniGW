#include "execution.hpp"

#include "cli.hpp"
#include "mpi_context.hpp"

#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gw {

std::string_view to_string(FrequencyParallelMode mode) noexcept {
    switch (mode) {
        case FrequencyParallelMode::Serial:
            return "serial";
        case FrequencyParallelMode::OpenMP:
            return "openmp";
        case FrequencyParallelMode::MPI:
            return "mpi";
    }
    return "unknown";
}

FrequencyParallelMode choose_frequency_parallel_mode(const Cli& cli, const MpiContext& mpi) {
    if (cli.frequency_parallel == "serial") {
        // Serial frequency mode means that all ranks, if any, execute the same
        // frequency loop collectively.  This is required by distributed dense
        // linear algebra backends such as ScaLAPACK/COSMA, where all MPI ranks
        // must enter the same BLACS/ScaLAPACK collectives in the same order.
        // Whether multi-rank serial mode is legal depends on the selected
        // backend and is checked in validate_backend_for_execution().
        (void)mpi;
        return FrequencyParallelMode::Serial;
    }
    if (cli.frequency_parallel == "mpi") {
        if (mpi.size() == 1) {
            throw std::runtime_error("--frequency-parallel mpi requires running with more than one MPI rank");
        }
#ifndef GW_ENABLE_MPI
        throw std::runtime_error("--frequency-parallel mpi requires a build configured with -DGW_ENABLE_MPI=ON");
#else
        return FrequencyParallelMode::MPI;
#endif
    }
    if (cli.frequency_parallel == "openmp") {
#ifndef GW_ENABLE_OPENMP_FREQUENCY_PARALLEL
        throw std::runtime_error("--frequency-parallel openmp requires -DGW_ENABLE_OPENMP_FREQUENCY_PARALLEL=ON");
#else
        return FrequencyParallelMode::OpenMP;
#endif
    }

    // auto: MPI frequency distribution if launched with multiple ranks; otherwise serial.
    if (mpi.size() > 1) {
#ifdef GW_ENABLE_MPI
        return FrequencyParallelMode::MPI;
#else
        throw std::runtime_error("Internal error: MPI size > 1 but GW_ENABLE_MPI is not defined");
#endif
    }
    return FrequencyParallelMode::Serial;
}

std::size_t frequency_workspace_replicas(FrequencyParallelMode mode) {
    if (mode == FrequencyParallelMode::OpenMP) {
#ifdef GW_ENABLE_OPENMP_FREQUENCY_PARALLEL
#ifdef _OPENMP
        return static_cast<std::size_t>(omp_get_max_threads());
#else
        return 1;
#endif
#else
        return 1;
#endif
    }
    return 1;
}

bool choose_openmp_kernel_loops(const Cli& cli, FrequencyParallelMode frequency_mode) {
    if (cli.kernel_parallel == "serial") {
        return false;
    }
    if (cli.kernel_parallel == "openmp") {
#ifndef GW_ENABLE_OPENMP_KERNEL_LOOPS
        throw std::runtime_error("--kernel-parallel openmp requires -DGW_ENABLE_OPENMP_KERNEL_LOOPS=ON");
#else
        return true;
#endif
    }

    // In auto mode, avoid multiplying MPI ranks by OpenMP kernel threads.
    if (frequency_mode == FrequencyParallelMode::MPI) {
        return false;
    }
#ifdef GW_ENABLE_OPENMP_KERNEL_LOOPS
    return true;
#else
    return false;
#endif
}

void configure_scalapack_frequency_groups(ExecutionPolicy& policy, const Cli& cli) {
    policy.scalapack_ranks_per_group = std::max<std::size_t>(1, cli.scalapack_ranks_per_group);

    if (policy.mpi_size <= 1) {
        policy.frequency_group_id = 0;
        policy.num_frequency_groups = 1;
        policy.frequency_group_rank = 0;
        policy.frequency_group_size = 1;
        policy.scalapack_ranks_per_group = 1;
        return;
    }

    if (policy.frequency_parallel_mode == FrequencyParallelMode::MPI) {
        const std::size_t ranks_per_group = std::min(policy.scalapack_ranks_per_group, policy.mpi_size);
        policy.scalapack_ranks_per_group = ranks_per_group;
        policy.num_frequency_groups = (policy.mpi_size + ranks_per_group - 1U) / ranks_per_group;
        policy.frequency_group_id = policy.mpi_rank / ranks_per_group;
        policy.frequency_group_rank = policy.mpi_rank % ranks_per_group;
        const std::size_t group_begin = policy.frequency_group_id * ranks_per_group;
        const std::size_t group_end = std::min(group_begin + ranks_per_group, policy.mpi_size);
        policy.frequency_group_size = group_end - group_begin;
        return;
    }

    // Serial frequency mode with a distributed backend means all ranks cooperate
    // on the same frequency point.  Therefore all ranks must be in a single
    // ScaLAPACK communicator group, irrespective of the CLI default.
    policy.frequency_group_id = 0;
    policy.num_frequency_groups = 1;
    policy.frequency_group_rank = policy.mpi_rank;
    policy.frequency_group_size = policy.mpi_size;
    policy.scalapack_ranks_per_group = policy.mpi_size;
}

} // namespace gw
