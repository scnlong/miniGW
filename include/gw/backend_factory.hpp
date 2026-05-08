#pragma once

#include "gw/cli.hpp"
#include "gw/execution.hpp"
#include "gw/linalg.hpp"

#include <memory>

namespace gw {

[[nodiscard]] std::shared_ptr<const linalg::Backend> make_local_linalg_backend(const Cli& cli);
void validate_backend_for_execution(const linalg::Backend& backend, const ExecutionPolicy& execution);

} // namespace gw
