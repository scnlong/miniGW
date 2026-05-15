#pragma once

#include "cli.hpp"
#include "execution.hpp"
#include "linalg/linalg.hpp"

#include <memory>

namespace gw {

[[nodiscard]] std::shared_ptr<const linalg::Backend> make_local_linalg_backend(const Cli& cli);
void validate_backend_for_execution(const linalg::Backend& backend, const ExecutionPolicy& execution);

} // namespace gw
