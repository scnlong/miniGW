#include "gw/linalg_cublas.hpp"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gw::linalg {
namespace {

[[nodiscard]] int checked_int(std::size_t value, const char* what) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("CUDA integer overflow for ") + what);
    }
    return static_cast<int>(value);
}

void check_cuda(cudaError_t status, const char* call) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(call) + " failed: " + cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const char* call) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(call) + " failed with cublasStatus_t=" + std::to_string(static_cast<int>(status)));
    }
}

void check_cusolver(cusolverStatus_t status, const char* call) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(call) + " failed with cusolverStatus_t=" + std::to_string(static_cast<int>(status)));
    }
}

[[nodiscard]] cuDoubleComplex to_cu(Complex z) noexcept {
    return make_cuDoubleComplex(z.real(), z.imag());
}

[[nodiscard]] Complex from_cu(cuDoubleComplex z) noexcept {
    return Complex{cuCreal(z), cuCimag(z)};
}

[[nodiscard]] std::size_t effective_rows(const MatrixComplex& a, MatrixTranspose trans) noexcept {
    return trans == MatrixTranspose::NoTranspose ? a.rows() : a.cols();
}

[[nodiscard]] std::size_t effective_cols(const MatrixComplex& a, MatrixTranspose trans) noexcept {
    return trans == MatrixTranspose::NoTranspose ? a.cols() : a.rows();
}

[[nodiscard]] Complex access(const MatrixComplex& a, std::size_t i, std::size_t j, MatrixTranspose trans) {
    switch (trans) {
        case MatrixTranspose::NoTranspose:
            return a(i, j);
        case MatrixTranspose::Transpose:
            return a(j, i);
        case MatrixTranspose::ConjugateTranspose:
            return std::conj(a(j, i));
    }
    throw std::runtime_error("unknown MatrixTranspose value");
}

[[nodiscard]] std::vector<cuDoubleComplex> materialize_column_major(const MatrixComplex& a,
                                                                    MatrixTranspose trans) {
    const std::size_t rows = effective_rows(a, trans);
    const std::size_t cols = effective_cols(a, trans);
    std::vector<cuDoubleComplex> out(rows * cols);
    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t i = 0; i < rows; ++i) {
            out[i + j * rows] = to_cu(access(a, i, j, trans));
        }
    }
    return out;
}

[[nodiscard]] MatrixComplex from_column_major(const std::vector<cuDoubleComplex>& a,
                                              std::size_t rows,
                                              std::size_t cols) {
    MatrixComplex out(rows, cols, Complex{0.0, 0.0});
    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t i = 0; i < rows; ++i) {
            out(i, j) = from_cu(a[i + j * rows]);
        }
    }
    return out;
}

class CudaBuffer {
public:
    CudaBuffer() = default;
    explicit CudaBuffer(std::size_t count) { allocate(count); }
    ~CudaBuffer() { reset(); }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void allocate(std::size_t count) {
        reset();
        count_ = count;
        if (count_ > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(cuDoubleComplex)), "cudaMalloc");
        }
    }

    void reset() noexcept {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    [[nodiscard]] cuDoubleComplex* get() noexcept { return ptr_; }
    [[nodiscard]] const cuDoubleComplex* get() const noexcept { return ptr_; }
    [[nodiscard]] std::size_t count() const noexcept { return count_; }

private:
    cuDoubleComplex* ptr_{nullptr};
    std::size_t count_{0};
};

class IntCudaBuffer {
public:
    IntCudaBuffer() = default;
    explicit IntCudaBuffer(std::size_t count) { allocate(count); }
    ~IntCudaBuffer() { reset(); }

    IntCudaBuffer(const IntCudaBuffer&) = delete;
    IntCudaBuffer& operator=(const IntCudaBuffer&) = delete;

    void allocate(std::size_t count) {
        reset();
        count_ = count;
        if (count_ > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(int)), "cudaMalloc");
        }
    }

    void reset() noexcept {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    [[nodiscard]] int* get() noexcept { return ptr_; }
    [[nodiscard]] const int* get() const noexcept { return ptr_; }

private:
    int* ptr_{nullptr};
    std::size_t count_{0};
};

[[nodiscard]] int device_info_value(int* d_info) {
    int info = 0;
    check_cuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(devInfo)");
    return info;
}

class CublasHandle {
public:
    CublasHandle() { check_cublas(cublasCreate(&handle_), "cublasCreate"); }
    ~CublasHandle() { if (handle_ != nullptr) { cublasDestroy(handle_); } }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    [[nodiscard]] cublasHandle_t get() const noexcept { return handle_; }
private:
    cublasHandle_t handle_{nullptr};
};

class CusolverHandle {
public:
    CusolverHandle() { check_cusolver(cusolverDnCreate(&handle_), "cusolverDnCreate"); }
    ~CusolverHandle() { if (handle_ != nullptr) { cusolverDnDestroy(handle_); } }
    CusolverHandle(const CusolverHandle&) = delete;
    CusolverHandle& operator=(const CusolverHandle&) = delete;
    [[nodiscard]] cusolverDnHandle_t get() const noexcept { return handle_; }
private:
    cusolverDnHandle_t handle_{nullptr};
};

} // namespace

std::string_view CublasBackend::name() const noexcept {
    return "cublas-cusolver-host-wrapper";
}

BackendCapabilities CublasBackend::capabilities() const noexcept {
    return BackendCapabilities{.family = BackendFamily::Device,
                               .thread_safe = false,
                               .uses_internal_threads = false,
                               .distributed_mpi = false,
                               .uses_device_memory = true};
}

MatrixComplex CublasBackend::inverse(MatrixComplex a) const {
    if (a.rows() != a.cols()) {
        throw std::runtime_error("CublasBackend::inverse: matrix must be square");
    }

    const int n = checked_int(a.rows(), "matrix dimension");
    const int lda = n;
    const int ldb = n;
    const int nrhs = n;

    std::vector<cuDoubleComplex> h_a = materialize_column_major(a, MatrixTranspose::NoTranspose);
    std::vector<cuDoubleComplex> h_b(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            h_b[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(n)] =
                to_cu(i == j ? Complex{1.0, 0.0} : Complex{0.0, 0.0});
        }
    }

    CusolverHandle solver;
    CudaBuffer d_a(h_a.size());
    CudaBuffer d_b(h_b.size());
    IntCudaBuffer d_ipiv(static_cast<std::size_t>(n));
    IntCudaBuffer d_info(1);

    check_cuda(cudaMemcpy(d_a.get(), h_a.data(), h_a.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "cudaMemcpy(A H2D)");
    check_cuda(cudaMemcpy(d_b.get(), h_b.data(), h_b.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "cudaMemcpy(B H2D)");

    int lwork = 0;
    check_cusolver(cusolverDnZgetrf_bufferSize(solver.get(), n, n, d_a.get(), lda, &lwork),
                   "cusolverDnZgetrf_bufferSize");
    CudaBuffer d_work(static_cast<std::size_t>(lwork));

    check_cusolver(cusolverDnZgetrf(solver.get(), n, n, d_a.get(), lda, d_work.get(), d_ipiv.get(), d_info.get()),
                   "cusolverDnZgetrf");
    const int getrf_info = device_info_value(d_info.get());
    if (getrf_info != 0) {
        throw std::runtime_error("cusolverDnZgetrf failed with info=" + std::to_string(getrf_info));
    }

    check_cusolver(cusolverDnZgetrs(solver.get(), CUBLAS_OP_N, n, nrhs, d_a.get(), lda, d_ipiv.get(), d_b.get(), ldb, d_info.get()),
                   "cusolverDnZgetrs");
    const int getrs_info = device_info_value(d_info.get());
    if (getrs_info != 0) {
        throw std::runtime_error("cusolverDnZgetrs failed with info=" + std::to_string(getrs_info));
    }

    check_cuda(cudaMemcpy(h_b.data(), d_b.get(), h_b.size() * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "cudaMemcpy(inverse D2H)");
    return from_column_major(h_b, a.rows(), a.cols());
}

MatrixComplex CublasBackend::gemm(const MatrixComplex& a,
                                  const MatrixComplex& b,
                                  MatrixTranspose trans_a,
                                  MatrixTranspose trans_b) const {
    const std::size_t m_size = effective_rows(a, trans_a);
    const std::size_t k_a = effective_cols(a, trans_a);
    const std::size_t k_b = effective_rows(b, trans_b);
    const std::size_t n_size = effective_cols(b, trans_b);
    if (k_a != k_b) {
        throw std::runtime_error("CublasBackend::gemm: incompatible matrix shapes");
    }

    const int m = checked_int(m_size, "m");
    const int n = checked_int(n_size, "n");
    const int k = checked_int(k_a, "k");
    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    std::vector<cuDoubleComplex> h_a = materialize_column_major(a, trans_a);
    std::vector<cuDoubleComplex> h_b = materialize_column_major(b, trans_b);
    std::vector<cuDoubleComplex> h_c(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));

    CublasHandle handle;
    CudaBuffer d_a(h_a.size());
    CudaBuffer d_b(h_b.size());
    CudaBuffer d_c(h_c.size());

    check_cuda(cudaMemcpy(d_a.get(), h_a.data(), h_a.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "cudaMemcpy(A H2D)");
    check_cuda(cudaMemcpy(d_b.get(), h_b.data(), h_b.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "cudaMemcpy(B H2D)");

    const cuDoubleComplex alpha = to_cu(Complex{1.0, 0.0});
    const cuDoubleComplex beta = to_cu(Complex{0.0, 0.0});
    check_cublas(cublasZgemm(handle.get(),
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             m,
                             n,
                             k,
                             &alpha,
                             d_a.get(),
                             lda,
                             d_b.get(),
                             ldb,
                             &beta,
                             d_c.get(),
                             ldc),
                 "cublasZgemm");

    check_cuda(cudaMemcpy(h_c.data(), d_c.get(), h_c.size() * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "cudaMemcpy(C D2H)");
    return from_column_major(h_c, m_size, n_size);
}

std::vector<Complex> CublasBackend::gemv(const MatrixComplex& a,
                                         const std::vector<double>& x,
                                         MatrixTranspose trans_a) const {
    const std::size_t m_size = effective_rows(a, trans_a);
    const std::size_t n_size = effective_cols(a, trans_a);
    if (x.size() != n_size) {
        throw std::runtime_error("CublasBackend::gemv: incompatible matrix/vector shapes");
    }

    const int m = checked_int(m_size, "m");
    const int n = checked_int(n_size, "n");
    const int lda = m;

    std::vector<cuDoubleComplex> h_a = materialize_column_major(a, trans_a);
    std::vector<cuDoubleComplex> h_x(x.size());
    std::vector<cuDoubleComplex> h_y(m_size);
    for (std::size_t i = 0; i < x.size(); ++i) {
        h_x[i] = to_cu(Complex{x[i], 0.0});
    }

    CublasHandle handle;
    CudaBuffer d_a(h_a.size());
    CudaBuffer d_x(h_x.size());
    CudaBuffer d_y(h_y.size());
    check_cuda(cudaMemcpy(d_a.get(), h_a.data(), h_a.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "cudaMemcpy(A H2D)");
    check_cuda(cudaMemcpy(d_x.get(), h_x.data(), h_x.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "cudaMemcpy(x H2D)");

    const cuDoubleComplex alpha = to_cu(Complex{1.0, 0.0});
    const cuDoubleComplex beta = to_cu(Complex{0.0, 0.0});
    check_cublas(cublasZgemv(handle.get(),
                             CUBLAS_OP_N,
                             m,
                             n,
                             &alpha,
                             d_a.get(),
                             lda,
                             d_x.get(),
                             1,
                             &beta,
                             d_y.get(),
                             1),
                 "cublasZgemv");

    check_cuda(cudaMemcpy(h_y.data(), d_y.get(), h_y.size() * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "cudaMemcpy(y D2H)");
    std::vector<Complex> y(h_y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] = from_cu(h_y[i]);
    }
    return y;
}

Complex CublasBackend::quadratic_form(const std::vector<double>& x, const MatrixComplex& a) const {
    if (a.rows() != a.cols() || x.size() != a.rows()) {
        throw std::runtime_error("CublasBackend::quadratic_form: incompatible shapes");
    }
    const std::vector<Complex> ax = gemv(a, x);
    Complex value{0.0, 0.0};
    for (std::size_t i = 0; i < x.size(); ++i) {
        value += x[i] * ax[i];
    }
    return value;
}

std::shared_ptr<const Backend> make_cublas_backend() {
    static const std::shared_ptr<const Backend> backend = std::make_shared<CublasBackend>();
    return backend;
}

} // namespace gw::linalg
