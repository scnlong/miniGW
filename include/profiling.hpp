#pragma once

#include "gw/gw.hpp"

#include <chrono>

namespace gw {

using ProfilingClock = std::chrono::steady_clock;

double elapsed_seconds(ProfilingClock::time_point start, ProfilingClock::time_point end);
void output_profiling_baseline(double read_input_seconds, const GwTimings& timings);

} // namespace gw
