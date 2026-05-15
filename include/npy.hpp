#pragma once

#include "types.hpp"

#include <string>
#include <vector>

namespace gw {

struct NpyArrayReal {
    std::vector<std::size_t> shape;
    std::vector<double> data;
};

NpyArrayReal read_npy_f64(const std::string& path);
std::vector<double> read_vector_npy_f64(const std::string& path);
MatrixReal read_matrix_npy_f64(const std::string& path);
Tensor4Real read_tensor4_npy_f64(const std::string& path);

} // namespace gw
