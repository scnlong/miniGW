#pragma once

#include "gw/linalg.hpp"

namespace gw::linalg {

// Semantic alias for the existing local, replicated host-memory backend family.
// Reference and BLAS/LAPACK implementations belong here. A simple correctness-only
// CUDA backend can also implement this interface, but a high-performance CUDA
// path should use a device-resident workspace instead of repeated host/device copies.
using LocalHostBackend = Backend;

} // namespace gw::linalg
