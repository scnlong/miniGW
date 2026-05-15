#include "workspace/device_screening_workspace.hpp"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <algorithm>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gw::workspace {
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
        throw std::runtime_error(std::string(call) + " failed with cublasStatus_t=" +
                                 std::to_string(static_cast<int>(status)));
    }
}

void check_cusolver(cusolverStatus_t status, const char* call) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(call) + " failed with cusolverStatus_t=" +
                                 std::to_string(static_cast<int>(status)));
    }
}

[[nodiscard]] cuDoubleComplex to_cu(Complex z) noexcept {
    return make_cuDoubleComplex(z.real(), z.imag());
}

[[nodiscard]] Complex from_cu(cuDoubleComplex z) noexcept {
    return Complex{cuCreal(z), cuCimag(z)};
}

class DeviceComplexBuffer {
public:
    DeviceComplexBuffer() = default;
    explicit DeviceComplexBuffer(std::size_t count) { allocate(count); }
    ~DeviceComplexBuffer() { reset(); }

    DeviceComplexBuffer(const DeviceComplexBuffer&) = delete;
    DeviceComplexBuffer& operator=(const DeviceComplexBuffer&) = delete;

    DeviceComplexBuffer(DeviceComplexBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    [[maybe_unused]] DeviceComplexBuffer& operator=(DeviceComplexBuffer&& other) noexcept {
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
        if (count == count_ && ptr_ != nullptr) {
            return;
        }
        reset();
        count_ = count;
        if (count_ > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(cuDoubleComplex)),
                       "cudaMalloc(complex)");
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
    [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(cuDoubleComplex); }

private:
    cuDoubleComplex* ptr_{nullptr};
    std::size_t count_{0};
};

class DeviceIntBuffer {
public:
    DeviceIntBuffer() = default;
    explicit DeviceIntBuffer(std::size_t count) { allocate(count); }
    ~DeviceIntBuffer() { reset(); }

    DeviceIntBuffer(const DeviceIntBuffer&) = delete;
    DeviceIntBuffer& operator=(const DeviceIntBuffer&) = delete;

    DeviceIntBuffer(DeviceIntBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    [[maybe_unused]] DeviceIntBuffer& operator=(DeviceIntBuffer&& other) noexcept {
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
        if (count == count_ && ptr_ != nullptr) {
            return;
        }
        reset();
        count_ = count;
        if (count_ > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(int)),
                       "cudaMalloc(int)");
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
    [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(int); }

private:
    int* ptr_{nullptr};
    std::size_t count_{0};
};

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

[[nodiscard]] int device_info_value(int* d_info) {
    int info = 0;
    check_cuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy(devInfo)");
    return info;
}

[[nodiscard]] std::vector<cuDoubleComplex> real_matrix_to_column_major_complex(const MatrixReal& a) {
    std::vector<cuDoubleComplex> out(a.rows() * a.cols());
    for (std::size_t j = 0; j < a.cols(); ++j) {
        for (std::size_t i = 0; i < a.rows(); ++i) {
            out[i + j * a.rows()] = make_cuDoubleComplex(a(i, j), 0.0);
        }
    }
    return out;
}

[[nodiscard]] std::vector<cuDoubleComplex> identity_column_major(int n) {
    std::vector<cuDoubleComplex> out(static_cast<std::size_t>(n) * static_cast<std::size_t>(n),
                                     make_cuDoubleComplex(0.0, 0.0));
    for (int i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * static_cast<std::size_t>(n)] =
            make_cuDoubleComplex(1.0, 0.0);
    }
    return out;
}

__global__ void build_epsilon_kernel(const cuDoubleComplex* __restrict__ v_ph,
                                     const cuDoubleComplex* __restrict__ pi0,
                                     cuDoubleComplex* __restrict__ epsilon,
                                     int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * n;
    if (idx >= total) {
        return;
    }
    const int row = idx % n;
    const int col = idx / n;
    const cuDoubleComplex v = v_ph[idx];
    const cuDoubleComplex p = pi0[col];
    cuDoubleComplex value = make_cuDoubleComplex(-(cuCreal(v) * cuCreal(p) - cuCimag(v) * cuCimag(p)),
                                                 -(cuCreal(v) * cuCimag(p) + cuCimag(v) * cuCreal(p)));
    if (row == col) {
        value = cuCadd(value, make_cuDoubleComplex(1.0, 0.0));
    }
    epsilon[idx] = value;
}

__global__ void subtract_identity_kernel(cuDoubleComplex* __restrict__ a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int idx = i + i * n;
        a[idx] = cuCsub(a[idx], make_cuDoubleComplex(1.0, 0.0));
    }
}

__global__ void quadratic_forms_kernel(const cuDoubleComplex* __restrict__ x,
                                       const cuDoubleComplex* __restrict__ y,
                                       cuDoubleComplex* __restrict__ out,
                                       int n,
                                       int nvec) {
    extern __shared__ cuDoubleComplex shared[];
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int row = tid; row < n; row += blockDim.x) {
        const int idx = row + col * n;
        const double xr = cuCreal(x[idx]);
        const cuDoubleComplex yy = y[idx];
        sum = cuCadd(sum, make_cuDoubleComplex(xr * cuCreal(yy), xr * cuCimag(yy)));
    }
    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = cuCadd(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0 && col < nvec) {
        out[col] = shared[0];
    }
}

[[nodiscard]] int reduction_threads(int n) noexcept {
    int threads = 256;
    if (n < 256) {
        threads = 1;
        while (threads < n) {
            threads <<= 1;
        }
    }
    return threads;
}

} // namespace

struct DeviceScreeningWorkspace::Impl {
    explicit Impl(const MatrixReal& v_ph) : n_(checked_int(v_ph.rows(), "n_ph")) {
        if (v_ph.rows() != v_ph.cols()) {
            throw std::runtime_error("DeviceScreeningWorkspace: V_ph must be square");
        }
        if (n_ == 0) {
            throw std::runtime_error("DeviceScreeningWorkspace: empty V_ph is not supported");
        }

        const std::size_t n2 = static_cast<std::size_t>(n_) * static_cast<std::size_t>(n_);
        d_v_ph.allocate(n2);
        d_inv_v.allocate(n2);
        d_epsilon.allocate(n2);
        d_inv_eps.allocate(n2);
        d_w_c.allocate(n2);
        d_pi0.allocate(static_cast<std::size_t>(n_));
        d_info.allocate(1);
        d_ipiv.allocate(static_cast<std::size_t>(n_));

        std::vector<cuDoubleComplex> h_v = real_matrix_to_column_major_complex(v_ph);
        check_cuda(cudaMemcpy(d_v_ph.get(), h_v.data(), h_v.size() * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(V_ph H2D)");

        compute_inverse_of_v();
    }

    void ensure_solver_workspace() {
        int lwork = 0;
        check_cusolver(cusolverDnZgetrf_bufferSize(solver.get(), n_, n_, d_epsilon.get(), n_, &lwork),
                       "cusolverDnZgetrf_bufferSize");
        d_work.allocate(static_cast<std::size_t>(lwork));
    }

    void factorize_and_solve_identity(DeviceComplexBuffer& d_a, DeviceComplexBuffer& d_b) {
        ensure_solver_workspace();
        check_cusolver(cusolverDnZgetrf(solver.get(), n_, n_, d_a.get(), n_, d_work.get(),
                                        d_ipiv.get(), d_info.get()),
                       "cusolverDnZgetrf");
        const int getrf_info = device_info_value(d_info.get());
        if (getrf_info != 0) {
            throw std::runtime_error("cusolverDnZgetrf failed with info=" + std::to_string(getrf_info));
        }
        check_cusolver(cusolverDnZgetrs(solver.get(), CUBLAS_OP_N, n_, n_, d_a.get(), n_, d_ipiv.get(),
                                        d_b.get(), n_, d_info.get()),
                       "cusolverDnZgetrs");
        const int getrs_info = device_info_value(d_info.get());
        if (getrs_info != 0) {
            throw std::runtime_error("cusolverDnZgetrs failed with info=" + std::to_string(getrs_info));
        }
    }

    void compute_inverse_of_v() {
        const std::size_t n2 = static_cast<std::size_t>(n_) * static_cast<std::size_t>(n_);
        check_cuda(cudaMemcpy(d_epsilon.get(), d_v_ph.get(), n2 * sizeof(cuDoubleComplex),
                              cudaMemcpyDeviceToDevice),
                   "cudaMemcpy(V_ph to factor buffer)");
        const std::vector<cuDoubleComplex> h_i = identity_column_major(n_);
        check_cuda(cudaMemcpy(d_inv_v.get(), h_i.data(), h_i.size() * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(I for inv_v)");
        factorize_and_solve_identity(d_epsilon, d_inv_v);
    }

    void compute_w_c(const std::vector<Complex>& pi0_diag) {
        if (pi0_diag.size() != static_cast<std::size_t>(n_)) {
            throw std::runtime_error("DeviceScreeningWorkspace::compute_w_c: inconsistent pi0 dimension");
        }
        std::vector<cuDoubleComplex> h_pi0(pi0_diag.size());
        for (std::size_t i = 0; i < pi0_diag.size(); ++i) {
            h_pi0[i] = to_cu(pi0_diag[i]);
        }
        check_cuda(cudaMemcpy(d_pi0.get(), h_pi0.data(), h_pi0.size() * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(pi0 H2D)");

        const int threads = 256;
        const int blocks = (n_ * n_ + threads - 1) / threads;
        build_epsilon_kernel<<<blocks, threads>>>(d_v_ph.get(), d_pi0.get(), d_epsilon.get(), n_);
        check_cuda(cudaGetLastError(), "build_epsilon_kernel");

        const std::vector<cuDoubleComplex> h_i = identity_column_major(n_);
        check_cuda(cudaMemcpy(d_inv_eps.get(), h_i.data(), h_i.size() * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(I for inv_eps)");
        factorize_and_solve_identity(d_epsilon, d_inv_eps);

        const int diag_threads = 256;
        const int diag_blocks = (n_ + diag_threads - 1) / diag_threads;
        subtract_identity_kernel<<<diag_blocks, diag_threads>>>(d_inv_eps.get(), n_);
        check_cuda(cudaGetLastError(), "subtract_identity_kernel");

        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        check_cublas(cublasZgemm(blas.get(),
                                 CUBLAS_OP_T,
                                 CUBLAS_OP_N,
                                 n_,
                                 n_,
                                 n_,
                                 &alpha,
                                 d_inv_eps.get(),
                                 n_,
                                 d_inv_v.get(),
                                 n_,
                                 &beta,
                                 d_w_c.get(),
                                 n_),
                     "cublasZgemm(W_c)");
    }

    std::vector<Complex> quadratic_forms_from_device_panel(const cuDoubleComplex* d_x, int nvec) {
        if (nvec == 0) {
            return {};
        }
        last_panel_width_ = static_cast<std::size_t>(nvec);
        const std::size_t panel_count = static_cast<std::size_t>(n_) * static_cast<std::size_t>(nvec);
        d_y_panel.allocate(panel_count);
        d_q_panel.allocate(static_cast<std::size_t>(nvec));

        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        check_cublas(cublasZgemm(blas.get(),
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 n_,
                                 nvec,
                                 n_,
                                 &alpha,
                                 d_w_c.get(),
                                 n_,
                                 d_x,
                                 n_,
                                 &beta,
                                 d_y_panel.get(),
                                 n_),
                     "cublasZgemm(W_c * X)");

        const int threads = reduction_threads(n_);
        quadratic_forms_kernel<<<nvec, threads, static_cast<std::size_t>(threads) * sizeof(cuDoubleComplex)>>>(
            d_x, d_y_panel.get(), d_q_panel.get(), n_, nvec);
        check_cuda(cudaGetLastError(), "quadratic_forms_kernel");

        std::vector<cuDoubleComplex> h_q(static_cast<std::size_t>(nvec));
        check_cuda(cudaMemcpy(h_q.data(), d_q_panel.get(), h_q.size() * sizeof(cuDoubleComplex),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(quadratic forms D2H)");
        std::vector<Complex> out(h_q.size());
        for (std::size_t i = 0; i < h_q.size(); ++i) {
            out[i] = from_cu(h_q[i]);
        }
        return out;
    }

    std::vector<Complex> quadratic_forms_panel(const MatrixReal& x_panel) {
        if (x_panel.rows() != static_cast<std::size_t>(n_)) {
            throw std::runtime_error("DeviceScreeningWorkspace::quadratic_forms_panel: inconsistent panel row count");
        }
        const int nvec = checked_int(x_panel.cols(), "panel width");
        if (nvec == 0) {
            return {};
        }
        const std::size_t panel_count = static_cast<std::size_t>(n_) * static_cast<std::size_t>(nvec);
        d_x_panel.allocate(panel_count);

        std::vector<cuDoubleComplex> h_x = real_matrix_to_column_major_complex(x_panel);
        check_cuda(cudaMemcpy(d_x_panel.get(), h_x.data(), h_x.size() * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(X panel H2D)");

        return quadratic_forms_from_device_panel(d_x_panel.get(), nvec);
    }

    std::vector<Complex> quadratic_forms_panel(const DeviceComplexPanelView& x_panel) {
        if (x_panel.rows() != static_cast<std::size_t>(n_)) {
            throw std::runtime_error("DeviceScreeningWorkspace::quadratic_forms_panel(device): inconsistent panel row count");
        }
        const int nvec = checked_int(x_panel.cols(), "device panel width");
        const auto* d_x = static_cast<const cuDoubleComplex*>(x_panel.device_data());
        if (nvec > 0 && d_x == nullptr) {
            throw std::runtime_error("DeviceScreeningWorkspace::quadratic_forms_panel(device): null device panel");
        }
        return quadratic_forms_from_device_panel(d_x, nvec);
    }

    [[nodiscard]] std::size_t estimated_device_bytes() const noexcept {
        return d_v_ph.bytes() + d_inv_v.bytes() + d_epsilon.bytes() + d_inv_eps.bytes() +
               d_w_c.bytes() + d_pi0.bytes() + d_work.bytes() + d_x_panel.bytes() +
               d_y_panel.bytes() + d_q_panel.bytes() + d_ipiv.bytes() + d_info.bytes();
    }

    int n_{0};
    CublasHandle blas;
    CusolverHandle solver;
    DeviceComplexBuffer d_v_ph;
    DeviceComplexBuffer d_inv_v;
    DeviceComplexBuffer d_epsilon;
    DeviceComplexBuffer d_inv_eps;
    DeviceComplexBuffer d_w_c;
    DeviceComplexBuffer d_pi0;
    DeviceComplexBuffer d_work;
    DeviceComplexBuffer d_x_panel;
    DeviceComplexBuffer d_y_panel;
    DeviceComplexBuffer d_q_panel;
    DeviceIntBuffer d_ipiv;
    DeviceIntBuffer d_info;
    std::size_t last_panel_width_{0};
};

DeviceScreeningWorkspace::DeviceScreeningWorkspace(const MatrixReal& v_ph)
    : impl_(std::make_unique<Impl>(v_ph)) {}

DeviceScreeningWorkspace::~DeviceScreeningWorkspace() = default;

DeviceScreeningWorkspace::DeviceScreeningWorkspace(DeviceScreeningWorkspace&&) noexcept = default;
DeviceScreeningWorkspace& DeviceScreeningWorkspace::operator=(DeviceScreeningWorkspace&&) noexcept = default;

std::size_t DeviceScreeningWorkspace::nph() const noexcept {
    return impl_ ? static_cast<std::size_t>(impl_->n_) : 0;
}

std::size_t DeviceScreeningWorkspace::last_panel_width() const noexcept {
    return impl_ ? impl_->last_panel_width_ : 0;
}

std::size_t DeviceScreeningWorkspace::estimated_device_bytes() const noexcept {
    return impl_ ? impl_->estimated_device_bytes() : 0;
}

matrix::DataOwnership DeviceScreeningWorkspace::ownership() const noexcept {
    matrix::DataOwnership ownership{};
    ownership.memory_space = matrix::MemorySpace::Device;
    ownership.distribution = matrix::Distribution::DeviceResident;
    int device = -1;
    cudaGetDevice(&device);
    ownership.device_id = device;
    return ownership;
}

void DeviceScreeningWorkspace::compute_w_c(const std::vector<Complex>& pi0_diag) {
    impl_->compute_w_c(pi0_diag);
}

std::vector<Complex> DeviceScreeningWorkspace::quadratic_forms_panel(const MatrixReal& x_panel) {
    return impl_->quadratic_forms_panel(x_panel);
}

std::vector<Complex> DeviceScreeningWorkspace::quadratic_forms_panel(const DeviceComplexPanelView& x_panel) {
    return impl_->quadratic_forms_panel(x_panel);
}

} // namespace gw::workspace
