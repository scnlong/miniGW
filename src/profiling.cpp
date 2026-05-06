#include "gw/profiling.hpp"

#include <iomanip>
#include <iostream>

namespace gw {

double elapsed_seconds(ProfilingClock::time_point start, ProfilingClock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

void output_profiling_baseline(double read_input_seconds, const GwTimings& timings) {
    const auto old_flags = std::cout.flags();
    const auto old_precision = std::cout.precision();

    std::cout << "\n--- Profiling Baseline ---\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::left << std::setw(22) << "read input:" << std::right << std::setw(8) << read_input_seconds << " s\n";
    std::cout << std::left << std::setw(22) << "total GW wall:" << std::right << std::setw(8) << timings.total_wall_seconds << " s\n";
    std::cout << std::left << std::setw(22) << "build mapping:" << std::right << std::setw(8) << timings.build_mapping_seconds << " s\n";
    std::cout << std::left << std::setw(22) << "exchange:" << std::right << std::setw(8) << timings.exchange_seconds << " s\n";
    std::cout << std::left << std::setw(22) << "inv(V_ph):" << std::right << std::setw(8) << timings.build_inv_v_seconds << " s\n";
    std::cout << std::left << std::setw(22) << "build Pi0 accum:" << std::right << std::setw(8) << timings.build_pi0_seconds << " s\n";
    std::cout << std::left << std::setw(22) << "build W_c accum:" << std::right << std::setw(8) << timings.invert_epsilon_seconds << " s\n";
    std::cout << std::left << std::setw(22) << "sigma_c accum:" << std::right << std::setw(8) << timings.sigma_c_seconds << " s\n";
    std::cout << std::left << std::setw(22) << "sigma_c wall:" << std::right << std::setw(8) << timings.sigma_c_wall_seconds << " s\n";
    std::cout << std::left << std::setw(22) << "MPI reduce:" << std::right << std::setw(8) << timings.mpi_reduce_seconds << " s\n";
    std::cout << std::left << std::setw(22) << "pade/output pade:" << std::right << std::setw(8) << timings.pade_seconds << " s\n";

    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
}

} // namespace gw
