#pragma once

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace gw {

using Complex = std::complex<double>;
constexpr double kHartreeToEv = 27.211386245988;
constexpr double kPi = 3.141592653589793238462643383279502884;

class MatrixReal {
public:
    MatrixReal() = default;
    MatrixReal(std::size_t rows, std::size_t cols, double value = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] const std::vector<double>& data() const noexcept { return data_; }
    [[nodiscard]] std::vector<double>& data() noexcept { return data_; }

    double& operator()(std::size_t i, std::size_t j) { return data_.at(i * cols_ + j); }
    const double& operator()(std::size_t i, std::size_t j) const { return data_.at(i * cols_ + j); }

private:
    std::size_t rows_{0};
    std::size_t cols_{0};
    std::vector<double> data_{};
};

class MatrixComplex {
public:
    MatrixComplex() = default;
    MatrixComplex(std::size_t rows, std::size_t cols, Complex value = {})
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] const std::vector<Complex>& data() const noexcept { return data_; }
    [[nodiscard]] std::vector<Complex>& data() noexcept { return data_; }

    Complex& operator()(std::size_t i, std::size_t j) { return data_.at(i * cols_ + j); }
    const Complex& operator()(std::size_t i, std::size_t j) const { return data_.at(i * cols_ + j); }

private:
    std::size_t rows_{0};
    std::size_t cols_{0};
    std::vector<Complex> data_{};
};

class Tensor3Real {
public:
    Tensor3Real() = default;
    Tensor3Real(std::size_t n0, std::size_t n1, std::size_t n2, double value = 0.0)
        : n0_(n0), n1_(n1), n2_(n2), data_(n0 * n1 * n2, value) {}

    [[nodiscard]] std::size_t dim0() const noexcept { return n0_; }
    [[nodiscard]] std::size_t dim1() const noexcept { return n1_; }
    [[nodiscard]] std::size_t dim2() const noexcept { return n2_; }

    double& operator()(std::size_t i, std::size_t j, std::size_t k) {
        return data_.at((i * n1_ + j) * n2_ + k);
    }
    const double& operator()(std::size_t i, std::size_t j, std::size_t k) const {
        return data_.at((i * n1_ + j) * n2_ + k);
    }

private:
    std::size_t n0_{0};
    std::size_t n1_{0};
    std::size_t n2_{0};
    std::vector<double> data_{};
};

class Tensor4Real {
public:
    Tensor4Real() = default;
    Tensor4Real(std::size_t n0, std::size_t n1, std::size_t n2, std::size_t n3, std::vector<double> data)
        : n0_(n0), n1_(n1), n2_(n2), n3_(n3), data_(std::move(data)) {
        if (data_.size() != n0_ * n1_ * n2_ * n3_) {
            throw std::runtime_error("Tensor4Real: data size does not match shape");
        }
    }

    [[nodiscard]] std::size_t dim0() const noexcept { return n0_; }
    [[nodiscard]] std::size_t dim1() const noexcept { return n1_; }
    [[nodiscard]] std::size_t dim2() const noexcept { return n2_; }
    [[nodiscard]] std::size_t dim3() const noexcept { return n3_; }

    double& operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
        return data_.at(((i * n1_ + j) * n2_ + k) * n3_ + l);
    }
    const double& operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const {
        return data_.at(((i * n1_ + j) * n2_ + k) * n3_ + l);
    }

private:
    std::size_t n0_{0};
    std::size_t n1_{0};
    std::size_t n2_{0};
    std::size_t n3_{0};
    std::vector<double> data_{};
};

} // namespace gw
