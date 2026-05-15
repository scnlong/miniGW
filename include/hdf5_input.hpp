#pragma once

#include "gw/gw.hpp"

#include <string>

namespace gw {

GwInput read_gw_input_hdf5(const std::string& path);

} // namespace gw
