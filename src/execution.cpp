#include "gw/execution.hpp"

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

} // namespace gw
