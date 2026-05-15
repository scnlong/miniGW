#pragma once

namespace gw::runtime {

void set_openmp_kernel_loops_enabled(bool enabled) noexcept;
bool openmp_kernel_loops_enabled() noexcept;

} // namespace gw::runtime
