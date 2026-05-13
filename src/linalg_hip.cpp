#include "gw/linalg_hip.hpp"

#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <hipsolver/hipsolver.h>

#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gw::linalg {
namespace {

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
#define GW_GPU_HD __host__ __device__ inline
#else
#define GW_GPU_HD inline
#endif

[[nodiscard]] GW_GPU_HD hipDoubleComplex make_gpu_complex(double real, double imag) noexcept {
    hipDoubleComplex z{};
    z.x = real;
    z.y = imag;
    return z;
}

[[nodiscard]] GW_GPU_HD double gpu_real(hipDoubleComplex z) noexcept { return z.x; }
[[nodiscard]] GW_GPU_HD double gpu_imag(hipDoubleComplex z) noexcept { return z.y; }

[[nodiscard]] GW_GPU_HD hipDoubleComplex gpu_add(hipDoubleComplex a,
                                                     hipDoubleComplex b) noexcept {
    return make_gpu_complex(a.x + b.x, a.y + b.y);
}

[[nodiscard]] GW_GPU_HD hipDoubleComplex gpu_sub(hipDoubleComplex a,
                                                     hipDoubleComplex b) noexcept {
    return make_gpu_complex(a.x - b.x, a.y - b.y);
}


[[nodiscard]] int checked_int(std::size_t value, const char* what) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("HIP/ROCm integer overflow for ") + what);
    }
    return static_cast<int>(value);
}

void check_hip(hipError_t status, const char* call) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(call) + " failed: " + hipGetErrorString(status));
    }
}

void check_hipblas(hipblasStatus_t status, const char* call) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(call) + " failed with hipblasStatus_t=" + std::to_string(static_cast<int>(status)));
    }
}

void check_hipsolver(hipsolverStatus_t status, const char* call) {
    if (status != HIPSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(call) + " failed with hipsolverStatus_t=" + std::to_string(static_cast<int>(status)));
    }
}

[[nodiscard]] hipDoubleComplex to_cu(Complex z) noexcept {
    return make_gpu_complex(z.real(), z.imag());
}

[[nodiscard]] Complex from_cu(hipDoubleComplex z) noexcept {
    return Complex{gpu_real(z), gpu_imag(z)};
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

[[nodiscard]] std::vector<hipDoubleComplex> materialize_column_major(const MatrixComplex& a,
                                                                    MatrixTranspose trans) {
    const std::size_t rows = effective_rows(a, trans);
    const std::size_t cols = effective_cols(a, trans);
    std::vector<hipDoubleComplex> out(rows * cols);
    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t i = 0; i < rows; ++i) {
            out[i + j * rows] = to_cu(access(a, i, j, trans));
        }
    }
    return out;
}

[[nodiscard]] MatrixComplex from_column_major(const std::vector<hipDoubleComplex>& a,
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

class HipBuffer {
public:
    HipBuffer() = default;
    explicit HipBuffer(std::size_t count) { allocate(count); }
    ~HipBuffer() { reset(); }

    HipBuffer(const HipBuffer&) = delete;
    HipBuffer& operator=(const HipBuffer&) = delete;

    HipBuffer(HipBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    HipBuffer& operator=(HipBuffer&& other) noexcept {
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
            check_hip(hipMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(hipDoubleComplex)), "hipMalloc");
        }
    }

    void reset() noexcept {
        if (ptr_ != nullptr) {
            (void)hipFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    [[nodiscard]] hipDoubleComplex* get() noexcept { return ptr_; }
    [[nodiscard]] const hipDoubleComplex* get() const noexcept { return ptr_; }
    [[nodiscard]] std::size_t count() const noexcept { return count_; }

private:
    hipDoubleComplex* ptr_{nullptr};
    std::size_t count_{0};
};

class IntHipBuffer {
public:
    IntHipBuffer() = default;
    explicit IntHipBuffer(std::size_t count) { allocate(count); }
    ~IntHipBuffer() { reset(); }

    IntHipBuffer(const IntHipBuffer&) = delete;
    IntHipBuffer& operator=(const IntHipBuffer&) = delete;

    void allocate(std::size_t count) {
        reset();
        count_ = count;
        if (count_ > 0) {
            check_hip(hipMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(int)), "hipMalloc");
        }
    }

    void reset() noexcept {
        if (ptr_ != nullptr) {
            (void)hipFree(ptr_);
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
    check_hip(hipMemcpy(&info, d_info, sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy(devInfo)");
    return info;
}

class HipHandle {
public:
    HipHandle() { check_hipblas(hipblasCreate(&handle_), "hipblasCreate"); }
    ~HipHandle() { if (handle_ != nullptr) { hipblasDestroy(handle_); } }
    HipHandle(const HipHandle&) = delete;
    HipHandle& operator=(const HipHandle&) = delete;
    [[nodiscard]] hipblasHandle_t get() const noexcept { return handle_; }
private:
    hipblasHandle_t handle_{nullptr};
};

class HipsolverHandle {
public:
    HipsolverHandle() { check_hipsolver(hipsolverCreate(&handle_), "hipsolverCreate"); }
    ~HipsolverHandle() { if (handle_ != nullptr) { hipsolverDestroy(handle_); } }
    HipsolverHandle(const HipsolverHandle&) = delete;
    HipsolverHandle& operator=(const HipsolverHandle&) = delete;
    [[nodiscard]] hipsolverHandle_t get() const noexcept { return handle_; }
private:
    hipsolverHandle_t handle_{nullptr};
};

} // namespace

std::string_view HipBackend::name() const noexcept {
    return "hipblas-hipsolver-host-wrapper";
}

BackendCapabilities HipBackend::capabilities() const noexcept {
    return BackendCapabilities{.family = BackendFamily::Device,
                               .thread_safe = false,
                               .uses_internal_threads = false,
                               .distributed_mpi = false,
                               .uses_device_memory = true};
}

MatrixComplex HipBackend::inverse(MatrixComplex a) const {
    if (a.rows() != a.cols()) {
        throw std::runtime_error("HipBackend::inverse: matrix must be square");
    }

    const int n = checked_int(a.rows(), "matrix dimension");
    const int lda = n;
    const int ldb = n;
    const int nrhs = n;

    std::vector<hipDoubleComplex> h_a = materialize_column_major(a, MatrixTranspose::NoTranspose);
    std::vector<hipDoubleComplex> h_b(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            h_b[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(n)] =
                to_cu(i == j ? Complex{1.0, 0.0} : Complex{0.0, 0.0});
        }
    }

    HipsolverHandle solver;
    HipBuffer d_a(h_a.size());
    HipBuffer d_b(h_b.size());
    IntHipBuffer d_ipiv(static_cast<std::size_t>(n));
    IntHipBuffer d_info(1);

    check_hip(hipMemcpy(d_a.get(), h_a.data(), h_a.size() * sizeof(hipDoubleComplex), hipMemcpyHostToDevice), "hipMemcpy(A H2D)");
    check_hip(hipMemcpy(d_b.get(), h_b.data(), h_b.size() * sizeof(hipDoubleComplex), hipMemcpyHostToDevice), "hipMemcpy(B H2D)");

    int getrf_lwork = 0;
    check_hipsolver(hipsolverZgetrf_bufferSize(solver.get(), n, n, d_a.get(), lda, &getrf_lwork),
                   "hipsolverZgetrf_bufferSize");

    int getrs_lwork = 0;
    check_hipsolver(hipsolverZgetrs_bufferSize(solver.get(), HIPSOLVER_OP_N, n, nrhs, d_a.get(), lda,
                                               d_ipiv.get(), d_b.get(), ldb, &getrs_lwork),
                   "hipsolverZgetrs_bufferSize");

    const int lwork = std::max({getrf_lwork, getrs_lwork, 1});
    HipBuffer d_work(static_cast<std::size_t>(lwork));

    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize before hipsolverZgetrf");
    check_hipsolver(hipsolverZgetrf(solver.get(), n, n, d_a.get(), lda, d_work.get(), lwork,
                                    d_ipiv.get(), d_info.get()),
                   "hipsolverZgetrf");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize after hipsolverZgetrf");
    const int getrf_info = device_info_value(d_info.get());
    if (getrf_info != 0) {
        throw std::runtime_error("hipsolverZgetrf failed with info=" + std::to_string(getrf_info));
    }

    check_hipsolver(hipsolverZgetrs(solver.get(), HIPSOLVER_OP_N, n, nrhs, d_a.get(), lda,
                                    d_ipiv.get(), d_b.get(), ldb, d_work.get(), lwork, d_info.get()),
                   "hipsolverZgetrs");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize after hipsolverZgetrs");
    const int getrs_info = device_info_value(d_info.get());
    if (getrs_info != 0) {
        throw std::runtime_error("hipsolverZgetrs failed with info=" + std::to_string(getrs_info));
    }

    check_hip(hipMemcpy(h_b.data(), d_b.get(), h_b.size() * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost), "hipMemcpy(inverse D2H)");
    return from_column_major(h_b, a.rows(), a.cols());
}

MatrixComplex HipBackend::gemm(const MatrixComplex& a,
                                  const MatrixComplex& b,
                                  MatrixTranspose trans_a,
                                  MatrixTranspose trans_b) const {
    const std::size_t m_size = effective_rows(a, trans_a);
    const std::size_t k_a = effective_cols(a, trans_a);
    const std::size_t k_b = effective_rows(b, trans_b);
    const std::size_t n_size = effective_cols(b, trans_b);
    if (k_a != k_b) {
        throw std::runtime_error("HipBackend::gemm: incompatible matrix shapes");
    }

    const int m = checked_int(m_size, "m");
    const int n = checked_int(n_size, "n");
    const int k = checked_int(k_a, "k");
    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    std::vector<hipDoubleComplex> h_a = materialize_column_major(a, trans_a);
    std::vector<hipDoubleComplex> h_b = materialize_column_major(b, trans_b);
    std::vector<hipDoubleComplex> h_c(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));

    HipHandle handle;
    HipBuffer d_a(h_a.size());
    HipBuffer d_b(h_b.size());
    HipBuffer d_c(h_c.size());

    check_hip(hipMemcpy(d_a.get(), h_a.data(), h_a.size() * sizeof(hipDoubleComplex), hipMemcpyHostToDevice), "hipMemcpy(A H2D)");
    check_hip(hipMemcpy(d_b.get(), h_b.data(), h_b.size() * sizeof(hipDoubleComplex), hipMemcpyHostToDevice), "hipMemcpy(B H2D)");

    const hipDoubleComplex alpha = to_cu(Complex{1.0, 0.0});
    const hipDoubleComplex beta = to_cu(Complex{0.0, 0.0});
    check_hipblas(hipblasZgemm(handle.get(),
                             HIPBLAS_OP_N,
                             HIPBLAS_OP_N,
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
                 "hipblasZgemm");

    check_hip(hipMemcpy(h_c.data(), d_c.get(), h_c.size() * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost), "hipMemcpy(C D2H)");
    return from_column_major(h_c, m_size, n_size);
}

std::vector<Complex> HipBackend::gemv(const MatrixComplex& a,
                                         const std::vector<double>& x,
                                         MatrixTranspose trans_a) const {
    const std::size_t m_size = effective_rows(a, trans_a);
    const std::size_t n_size = effective_cols(a, trans_a);
    if (x.size() != n_size) {
        throw std::runtime_error("HipBackend::gemv: incompatible matrix/vector shapes");
    }

    const int m = checked_int(m_size, "m");
    const int n = checked_int(n_size, "n");
    const int lda = m;

    std::vector<hipDoubleComplex> h_a = materialize_column_major(a, trans_a);
    std::vector<hipDoubleComplex> h_x(x.size());
    std::vector<hipDoubleComplex> h_y(m_size);
    for (std::size_t i = 0; i < x.size(); ++i) {
        h_x[i] = to_cu(Complex{x[i], 0.0});
    }

    HipHandle handle;
    HipBuffer d_a(h_a.size());
    HipBuffer d_x(h_x.size());
    HipBuffer d_y(h_y.size());
    check_hip(hipMemcpy(d_a.get(), h_a.data(), h_a.size() * sizeof(hipDoubleComplex), hipMemcpyHostToDevice), "hipMemcpy(A H2D)");
    check_hip(hipMemcpy(d_x.get(), h_x.data(), h_x.size() * sizeof(hipDoubleComplex), hipMemcpyHostToDevice), "hipMemcpy(x H2D)");

    const hipDoubleComplex alpha = to_cu(Complex{1.0, 0.0});
    const hipDoubleComplex beta = to_cu(Complex{0.0, 0.0});
    check_hipblas(hipblasZgemv(handle.get(),
                             HIPBLAS_OP_N,
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
                 "hipblasZgemv");

    check_hip(hipMemcpy(h_y.data(), d_y.get(), h_y.size() * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost), "hipMemcpy(y D2H)");
    std::vector<Complex> y(h_y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] = from_cu(h_y[i]);
    }
    return y;
}

Complex HipBackend::quadratic_form(const std::vector<double>& x, const MatrixComplex& a) const {
    if (a.rows() != a.cols() || x.size() != a.rows()) {
        throw std::runtime_error("HipBackend::quadratic_form: incompatible shapes");
    }
    const std::vector<Complex> ax = gemv(a, x);
    Complex value{0.0, 0.0};
    for (std::size_t i = 0; i < x.size(); ++i) {
        value += x[i] * ax[i];
    }
    return value;
}


int hip_device_count() {
    int count = 0;
    check_hip(hipGetDeviceCount(&count), "hipGetDeviceCount");
    if (count <= 0) {
        throw std::runtime_error("No HIP/ROCm devices are visible to this process");
    }
    return count;
}

int select_hip_device_for_local_rank(std::size_t mpi_local_rank, std::size_t tasks_per_gpu) {
    if (tasks_per_gpu == 0) {
        throw std::runtime_error("--tasks-per-gpu must be positive");
    }
    const int count = hip_device_count();
    return static_cast<int>((mpi_local_rank / tasks_per_gpu) % static_cast<std::size_t>(count));
}

int set_hip_device_for_local_rank(std::size_t mpi_local_rank, std::size_t tasks_per_gpu) {
    const int device = select_hip_device_for_local_rank(mpi_local_rank, tasks_per_gpu);
    check_hip(hipSetDevice(device), "hipSetDevice");
    return device;
}

std::shared_ptr<const Backend> make_hip_backend() {
    static const std::shared_ptr<const Backend> backend = std::make_shared<HipBackend>();
    return backend;
}

} // namespace gw::linalg
