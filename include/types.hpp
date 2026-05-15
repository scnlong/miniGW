#pragma once

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gw {

using Complex = std::complex<double>;
constexpr double kHartreeToEv = 27.211386245988;
constexpr double kPi = 3.141592653589793238462643383279502884;

template <typename T>
class Matrix {
public:
    Matrix() = default;

    Matrix(std::size_t rows, std::size_t cols, T value = {})
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}

    Matrix(std::size_t rows, std::size_t cols, std::vector<T> data)
        : rows_(rows), cols_(cols), data_(std::move(data)) {
        if (data_.size() != rows_ * cols_) {
            throw std::runtime_error("Matrix: data size does not match shape");
        }
    }

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }

    [[nodiscard]] const std::vector<T>& data() const noexcept { return data_; }
    [[nodiscard]] std::vector<T>& data() noexcept { return data_; }

    T& operator()(std::size_t i, std::size_t j) {
#ifndef NDEBUG
        check_bounds(i, j);
#endif
        return data_[index(i, j)];
    }

    const T& operator()(std::size_t i, std::size_t j) const {
#ifndef NDEBUG
        check_bounds(i, j);
#endif
        return data_[index(i, j)];
    }

private:
    std::size_t rows_{0};
    std::size_t cols_{0};
    std::vector<T> data_{};

    [[nodiscard]] std::size_t index(std::size_t i, std::size_t j) const noexcept {
        // Row-major / C-order layout: A[i, j]
        return i * cols_ + j;
    }

    void check_bounds(std::size_t i, std::size_t j) const {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range("Matrix index out of range");
        }
    }
};

template <typename T>
class Tensor3 {
public:
    Tensor3() = default;

    Tensor3(std::size_t n0, std::size_t n1, std::size_t n2, T value = {})
        : n0_(n0), n1_(n1), n2_(n2), data_(n0 * n1 * n2, value) {}

    Tensor3(std::size_t n0, std::size_t n1, std::size_t n2, std::vector<T> data)
        : n0_(n0), n1_(n1), n2_(n2), data_(std::move(data)) {
        if (data_.size() != n0_ * n1_ * n2_) {
            throw std::runtime_error("Tensor3: data size does not match shape");
        }
    }

    [[nodiscard]] std::size_t dim0() const noexcept { return n0_; }
    [[nodiscard]] std::size_t dim1() const noexcept { return n1_; }
    [[nodiscard]] std::size_t dim2() const noexcept { return n2_; }
    [[nodiscard]] std::size_t extent(std::size_t dim) const {
        switch (dim) {
            case 0: return n0_;
            case 1: return n1_;
            case 2: return n2_;
            default: throw std::out_of_range("Tensor3 dimension out of range");
        }
    }

    [[nodiscard]] const std::vector<T>& data() const noexcept { return data_; }
    [[nodiscard]] std::vector<T>& data() noexcept { return data_; }

    T& operator()(std::size_t i, std::size_t j, std::size_t k) {
#ifndef NDEBUG
        check_bounds(i, j, k);
#endif
        return data_[index(i, j, k)];
    }

    const T& operator()(std::size_t i, std::size_t j, std::size_t k) const {
#ifndef NDEBUG
        check_bounds(i, j, k);
#endif
        return data_[index(i, j, k)];
    }

private:
    std::size_t n0_{0};
    std::size_t n1_{0};
    std::size_t n2_{0};
    std::vector<T> data_{};

    [[nodiscard]] std::size_t index(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return (i * n1_ + j) * n2_ + k;
    }

    void check_bounds(std::size_t i, std::size_t j, std::size_t k) const {
        if (i >= n0_ || j >= n1_ || k >= n2_) {
            throw std::out_of_range("Tensor3 index out of range");
        }
    }
};

template <typename T>
class Tensor4 {
public:
    Tensor4() = default;

    Tensor4(std::size_t n0, std::size_t n1, std::size_t n2, std::size_t n3, T value = {})
        : n0_(n0), n1_(n1), n2_(n2), n3_(n3), data_(n0 * n1 * n2 * n3, value) {}

    Tensor4(std::size_t n0, std::size_t n1, std::size_t n2, std::size_t n3, std::vector<T> data)
        : n0_(n0), n1_(n1), n2_(n2), n3_(n3), data_(std::move(data)) {
        if (data_.size() != n0_ * n1_ * n2_ * n3_) {
            throw std::runtime_error("Tensor4: data size does not match shape");
        }
    }

    [[nodiscard]] std::size_t dim0() const noexcept { return n0_; }
    [[nodiscard]] std::size_t dim1() const noexcept { return n1_; }
    [[nodiscard]] std::size_t dim2() const noexcept { return n2_; }
    [[nodiscard]] std::size_t dim3() const noexcept { return n3_; }
    [[nodiscard]] std::size_t extent(std::size_t dim) const {
        switch (dim) {
            case 0: return n0_;
            case 1: return n1_;
            case 2: return n2_;
            case 3: return n3_;
            default: throw std::out_of_range("Tensor4 dimension out of range");
        }
    }

    [[nodiscard]] const std::vector<T>& data() const noexcept { return data_; }
    [[nodiscard]] std::vector<T>& data() noexcept { return data_; }

    T& operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
#ifndef NDEBUG
        check_bounds(i, j, k, l);
#endif
        return data_[index(i, j, k, l)];
    }

    const T& operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const {
#ifndef NDEBUG
        check_bounds(i, j, k, l);
#endif
        return data_[index(i, j, k, l)];
    }

private:
    std::size_t n0_{0};
    std::size_t n1_{0};
    std::size_t n2_{0};
    std::size_t n3_{0};
    std::vector<T> data_{};

    [[nodiscard]] std::size_t index(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const noexcept {
        // Row-major / C-order layout: T[i, j, k, l]
        return (((i * n1_ + j) * n2_ + k) * n3_ + l);
    }

    void check_bounds(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const {
        if (i >= n0_ || j >= n1_ || k >= n2_ || l >= n3_) {
            throw std::out_of_range("Tensor4 index out of range");
        }
    }
};

using MatrixReal = Matrix<double>;
using MatrixComplex = Matrix<Complex>;
using Tensor3Real = Tensor3<double>;
using Tensor4Real = Tensor4<double>;

} // namespace gw
