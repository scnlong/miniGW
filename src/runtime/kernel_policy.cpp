#include "runtime/kernel_policy.hpp"

#include <atomic>

namespace gw::runtime {
namespace {

#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
std::atomic_bool g_openmp_kernel_loops_enabled{true};
#endif

} // namespace

void set_openmp_kernel_loops_enabled(bool enabled) noexcept {
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
    g_openmp_kernel_loops_enabled.store(enabled, std::memory_order_relaxed);
#else
    (void)enabled;
#endif
}

bool openmp_kernel_loops_enabled() noexcept {
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
    return g_openmp_kernel_loops_enabled.load(std::memory_order_relaxed);
#else
    return false;
#endif
}

} // namespace gw::runtime
