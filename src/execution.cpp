#include "gw/execution.hpp"

#include "gw/cli.hpp"
#include "gw/mpi_context.hpp"

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
        if (mpi.size() > 1) {
            throw std::runtime_error("--frequency-parallel serial must not be launched with multiple MPI ranks; run without mpirun or use --frequency-parallel mpi");
        }
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

} // namespace gw
