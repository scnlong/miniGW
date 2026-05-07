#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace gw {

struct Cli {
    std::string input_dir{"."};
    std::string output_dir{"./gw_output"};
    std::size_t freq_points{200};
    std::size_t pade_params{16};
    std::optional<std::size_t> selected_state_1based{5};
    double eta{0.0};
    std::string linalg_backend{"blas-lapack"};
    std::string frequency_parallel{"auto"};
    std::string kernel_parallel{"auto"};
    bool print_memory_footprint{true};
};

Cli parse_cli(int argc, char** argv);
std::string join_path(const std::string& dir, const std::string& file);

} // namespace gw
