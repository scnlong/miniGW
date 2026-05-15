#include "npy.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace gw {
namespace {

std::string read_exact(std::ifstream& in, std::size_t n) {
    std::string s(n, '\0');
    in.read(s.data(), static_cast<std::streamsize>(n));
    if (!in) {
        throw std::runtime_error("Unexpected end of .npy file");
    }
    return s;
}

std::vector<std::size_t> parse_shape(const std::string& header) {
    const std::regex shape_regex(R"('shape'\s*:\s*\(([^\)]*)\))");
    std::smatch match;
    if (!std::regex_search(header, match, shape_regex)) {
        throw std::runtime_error("Could not parse .npy shape");
    }
    std::string body = match[1].str();
    std::vector<std::size_t> shape;
    std::stringstream ss(body);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char c) { return std::isspace(c); }), token.end());
        if (!token.empty()) {
            shape.push_back(static_cast<std::size_t>(std::stoull(token)));
        }
    }
    if (shape.empty()) {
        throw std::runtime_error("Parsed empty .npy shape");
    }
    return shape;
}

bool contains_fortran_true(const std::string& header) {
    return header.find("'fortran_order': True") != std::string::npos ||
           header.find("\"fortran_order\": True") != std::string::npos;
}

bool is_f64_descr(const std::string& header) {
    return header.find("'descr': '<f8'") != std::string::npos ||
           header.find("'descr': '|f8'") != std::string::npos ||
           header.find("\"descr\": \"<f8\"") != std::string::npos ||
           header.find("\"descr\": \"|f8\"") != std::string::npos;
}

std::size_t product(const std::vector<std::size_t>& shape) {
    std::size_t n = 1;
    for (const auto dim : shape) {
        n *= dim;
    }
    return n;
}

} // namespace

NpyArrayReal read_npy_f64(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open .npy file: " + path);
    }

    const std::string magic = read_exact(in, 6);
    if (magic != "\x93NUMPY") {
        throw std::runtime_error("Invalid .npy magic in file: " + path);
    }
    const unsigned char major = static_cast<unsigned char>(read_exact(in, 1)[0]);
    (void)read_exact(in, 1); // minor

    std::uint32_t header_len = 0;
    if (major == 1) {
        const std::string len_bytes = read_exact(in, 2);
        header_len = static_cast<unsigned char>(len_bytes[0]) |
                     (static_cast<std::uint32_t>(static_cast<unsigned char>(len_bytes[1])) << 8U);
    } else if (major == 2 || major == 3) {
        const std::string len_bytes = read_exact(in, 4);
        header_len = static_cast<unsigned char>(len_bytes[0]) |
                     (static_cast<std::uint32_t>(static_cast<unsigned char>(len_bytes[1])) << 8U) |
                     (static_cast<std::uint32_t>(static_cast<unsigned char>(len_bytes[2])) << 16U) |
                     (static_cast<std::uint32_t>(static_cast<unsigned char>(len_bytes[3])) << 24U);
    } else {
        throw std::runtime_error("Unsupported .npy version in file: " + path);
    }

    const std::string header = read_exact(in, header_len);
    if (!is_f64_descr(header)) {
        throw std::runtime_error("Only little-endian float64 .npy arrays are supported: " + path);
    }
    if (contains_fortran_true(header)) {
        throw std::runtime_error("Fortran-order .npy arrays are not supported in this minimal reader: " + path);
    }

    NpyArrayReal array;
    array.shape = parse_shape(header);
    const std::size_t n = product(array.shape);
    array.data.resize(n);
    in.read(reinterpret_cast<char*>(array.data.data()), static_cast<std::streamsize>(n * sizeof(double)));
    if (!in) {
        throw std::runtime_error("Could not read .npy payload: " + path);
    }
    return array;
}

std::vector<double> read_vector_npy_f64(const std::string& path) {
    auto arr = read_npy_f64(path);
    if (arr.shape.size() != 1) {
        throw std::runtime_error("Expected a 1D .npy vector: " + path);
    }
    return std::move(arr.data);
}

MatrixReal read_matrix_npy_f64(const std::string& path) {
    auto arr = read_npy_f64(path);
    if (arr.shape.size() != 2) {
        throw std::runtime_error("Expected a 2D .npy matrix: " + path);
    }
    MatrixReal matrix(arr.shape[0], arr.shape[1], 0.0);
    matrix.data() = std::move(arr.data);
    return matrix;
}

Tensor4Real read_tensor4_npy_f64(const std::string& path) {
    auto arr = read_npy_f64(path);
    if (arr.shape.size() != 4) {
        throw std::runtime_error("Expected a 4D .npy tensor: " + path);
    }
    return Tensor4Real(arr.shape[0], arr.shape[1], arr.shape[2], arr.shape[3], std::move(arr.data));
}

} // namespace gw
