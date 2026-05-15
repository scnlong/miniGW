#pragma once

#include "linalg/linalg.hpp"
#include "types.hpp"

#include <string_view>

namespace gw::linalg {

class DistributedBackend {
public:
    virtual ~DistributedBackend() = default;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    // Placeholder interface for future ScaLAPACK/COSMA implementations.  These
    // routines need a distributed matrix descriptor rather than MatrixComplex.
    // COSMA should be modeled as a distributed GEMM provider, typically paired
    // with ScaLAPACK/SLATE for factorization and solve.
};

} // namespace gw::linalg
