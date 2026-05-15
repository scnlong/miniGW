#pragma once

#include "types.hpp"
#include "linalg/linalg.hpp"

#include <string_view>

namespace gw::linalg {

class DeviceBackend {
public:
    virtual ~DeviceBackend() = default;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    // Placeholder interface for future device-resident implementations.  A real
    // cuBLAS/cuSolver backend should accept DeviceMatrix objects and explicit
    // workspaces/streams, not replicated host MatrixComplex objects.
};

} // namespace gw::linalg
