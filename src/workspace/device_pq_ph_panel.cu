#include "gw/workspace/device_pq_ph_panel.hpp"

#include <cuda_runtime_api.h>
#include <cuComplex.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
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

template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
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
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)), "cudaMalloc(DeviceBuffer)");
        }
    }

    void reset() noexcept {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    [[nodiscard]] T* get() noexcept { return ptr_; }
    [[nodiscard]] const T* get() const noexcept { return ptr_; }
    [[nodiscard]] std::size_t count() const noexcept { return count_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }

private:
    T* ptr_{nullptr};
    std::size_t count_{0};
};

template <class T>
class PinnedHostBuffer {
public:
    PinnedHostBuffer() = default;
    explicit PinnedHostBuffer(std::size_t count) { allocate(count); }
    ~PinnedHostBuffer() { reset(); }

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
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
            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T), cudaHostAllocDefault),
                       "cudaHostAlloc(PinnedHostBuffer)");
        }
    }

    void reset() noexcept {
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    [[nodiscard]] T* get() noexcept { return ptr_; }
    [[nodiscard]] const T* get() const noexcept { return ptr_; }
    [[nodiscard]] std::size_t count() const noexcept { return count_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }

private:
    T* ptr_{nullptr};
    std::size_t count_{0};
};

__global__ void build_pq_panel_kernel(const double* __restrict__ eri,
                                      const int* __restrict__ occ,
                                      const int* __restrict__ virt,
                                      cuDoubleComplex* __restrict__ panel,
                                      int nmo,
                                      int nph,
                                      int p_index,
                                      int k_begin,
                                      int width) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nph * width;
    if (idx >= total) {
        return;
    }

    const int ph = idx % nph;
    const int kk = idx / nph;
    const int k_index = k_begin + kk;
    const int i_occ = occ[ph];
    const int a_mo = virt[ph];

    const long long n = static_cast<long long>(nmo);
    const long long eri_idx = (((static_cast<long long>(p_index) * n + k_index) * n + i_occ) * n + a_mo);
    const double value = eri[eri_idx];
    panel[static_cast<long long>(ph) + static_cast<long long>(kk) * nph] = make_cuDoubleComplex(value, 0.0);
}

[[nodiscard]] DevicePqPhPanelStorageMode choose_mode(DevicePqPhPanelStorageMode requested,
                                                     std::size_t eri_bytes,
                                                     std::size_t panel_bytes) {
    if (requested != DevicePqPhPanelStorageMode::Auto) {
        return requested;
    }

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        cudaGetLastError();
        return DevicePqPhPanelStorageMode::StreamingPanel;
    }

    // Keep this deliberately conservative because DeviceScreeningWorkspace also
    // holds several n_ph^2 complex matrices and cuSolver workspace.  Full ERI
    // residency is only selected when ERI is comparatively cheap.
    const std::size_t budget = static_cast<std::size_t>(0.25 * static_cast<double>(free_bytes));
    return (eri_bytes + panel_bytes < budget)
               ? DevicePqPhPanelStorageMode::FullEriResident
               : DevicePqPhPanelStorageMode::StreamingPanel;
}

} // namespace

std::string_view to_string(DevicePqPhPanelStorageMode mode) noexcept {
    switch (mode) {
        case DevicePqPhPanelStorageMode::Auto: return "auto";
        case DevicePqPhPanelStorageMode::FullEriResident: return "full-eri-resident";
        case DevicePqPhPanelStorageMode::StreamingPanel: return "streaming-panel";
    }
    return "unknown";
}

struct DevicePqPhPanelView::Impl {
    Impl(const MolecularIntegrals& integrals,
         const ParticleHoleBasis& ph_basis,
         std::size_t max_panel_width_in,
         DevicePqPhPanelStorageMode requested_mode)
        : eri_host(&integrals.eri_tensor()),
          nmo(checked_int(integrals.nmo(), "nmo")),
          nph(checked_int(ph_basis.size(), "n_ph")),
          max_panel_width(checked_int(std::max<std::size_t>(1, max_panel_width_in), "max panel width")) {
        if (nmo <= 0 || nph <= 0) {
            throw std::runtime_error("DevicePqPhPanelView: empty ERI/ph basis is not supported");
        }

        const Tensor4Real& eri = *eri_host;
        if (eri.dim0() != static_cast<std::size_t>(nmo) ||
            eri.dim1() != static_cast<std::size_t>(nmo) ||
            eri.dim2() != static_cast<std::size_t>(nmo) ||
            eri.dim3() != static_cast<std::size_t>(nmo)) {
            throw std::runtime_error("DevicePqPhPanelView: ERI tensor shape mismatch");
        }

        std::vector<int> h_occ(static_cast<std::size_t>(nph));
        std::vector<int> h_virt(static_cast<std::size_t>(nph));
        for (std::size_t ph = 0; ph < ph_basis.size(); ++ph) {
            h_occ[ph] = checked_int(ph_basis[ph].i_occ, "ph occ index");
            h_virt[ph] = checked_int(ph_basis[ph].a_mo, "ph virt index");
        }
        occ_host = h_occ;
        virt_host = h_virt;
        d_occ.allocate(h_occ.size());
        d_virt.allocate(h_virt.size());
        check_cuda(cudaMemcpy(d_occ.get(), h_occ.data(), h_occ.size() * sizeof(int), cudaMemcpyHostToDevice),
                   "cudaMemcpy(ph occ H2D)");
        check_cuda(cudaMemcpy(d_virt.get(), h_virt.data(), h_virt.size() * sizeof(int), cudaMemcpyHostToDevice),
                   "cudaMemcpy(ph virt H2D)");

        d_panel.allocate(static_cast<std::size_t>(nph) * static_cast<std::size_t>(max_panel_width));

        const std::size_t eri_bytes = eri.data().size() * sizeof(double);
        const std::size_t panel_bytes = static_cast<std::size_t>(nph) * static_cast<std::size_t>(max_panel_width) * sizeof(cuDoubleComplex);
        mode = choose_mode(requested_mode, eri_bytes, panel_bytes);

        if (mode == DevicePqPhPanelStorageMode::FullEriResident) {
            d_eri.allocate(eri.data().size());
            check_cuda(cudaMemcpy(d_eri.get(), eri.data().data(), eri.data().size() * sizeof(double), cudaMemcpyHostToDevice),
                       "cudaMemcpy(ERI H2D)");
        } else {
            h_panel.allocate(static_cast<std::size_t>(nph) * static_cast<std::size_t>(max_panel_width));
        }
    }

    void ensure_panel_capacity(std::size_t width_in) {
        if (width_in <= static_cast<std::size_t>(max_panel_width)) {
            return;
        }
        max_panel_width = checked_int(width_in, "expanded max panel width");
        const std::size_t count = static_cast<std::size_t>(nph) * static_cast<std::size_t>(max_panel_width);
        d_panel.allocate(count);
        if (mode == DevicePqPhPanelStorageMode::StreamingPanel) {
            h_panel.allocate(count);
        }
    }

    void fill_panel_streaming(std::size_t p_index, std::size_t k_begin, std::size_t width) {
        const Tensor4Real& eri = *eri_host;
        cuDoubleComplex* panel = h_panel.get();
        for (std::size_t kk = 0; kk < width; ++kk) {
            const std::size_t k_index = k_begin + kk;
            for (std::size_t ph = 0; ph < static_cast<std::size_t>(nph); ++ph) {
                const std::size_t i_occ = static_cast<std::size_t>(occ_host[ph]);
                const std::size_t a_mo = static_cast<std::size_t>(virt_host[ph]);
                const double value = eri(p_index, k_index, i_occ, a_mo);
                panel[ph + kk * static_cast<std::size_t>(nph)] = make_cuDoubleComplex(value, 0.0);
            }
        }
        check_cuda(cudaMemcpy(d_panel.get(), h_panel.get(),
                              static_cast<std::size_t>(nph) * width * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(pq panel H2D streaming)");
    }

    DeviceComplexPanelView fill_panel(std::size_t p_index, std::size_t k_begin, std::size_t width_in) {
        if (p_index >= static_cast<std::size_t>(nmo)) {
            throw std::runtime_error("DevicePqPhPanelView::fill_panel: p index out of range");
        }
        if (k_begin > static_cast<std::size_t>(nmo) || width_in > static_cast<std::size_t>(nmo) - k_begin) {
            throw std::runtime_error("DevicePqPhPanelView::fill_panel: k panel exceeds nmo");
        }
        if (width_in == 0) {
            last_panel_width = 0;
            return DeviceComplexPanelView{d_panel.get(), static_cast<std::size_t>(nph), 0};
        }
        ensure_panel_capacity(width_in);

        if (mode == DevicePqPhPanelStorageMode::FullEriResident) {
            const int width = checked_int(width_in, "panel width");
            const int threads = 256;
            const int total = nph * width;
            const int blocks = (total + threads - 1) / threads;
            build_pq_panel_kernel<<<blocks, threads>>>(d_eri.get(), d_occ.get(), d_virt.get(), d_panel.get(),
                                                       nmo, nph,
                                                       checked_int(p_index, "p index"),
                                                       checked_int(k_begin, "k begin"),
                                                       width);
            check_cuda(cudaGetLastError(), "build_pq_panel_kernel");
        } else {
            fill_panel_streaming(p_index, k_begin, width_in);
        }

        last_panel_width = width_in;
        return DeviceComplexPanelView{d_panel.get(), static_cast<std::size_t>(nph), width_in};
    }

    [[nodiscard]] std::size_t estimated_device_bytes() const noexcept {
        return d_eri.bytes() + d_occ.bytes() + d_virt.bytes() + d_panel.bytes();
    }

    [[nodiscard]] std::size_t estimated_host_pinned_bytes() const noexcept {
        return h_panel.bytes();
    }

    const Tensor4Real* eri_host{nullptr};
    std::vector<int> occ_host{};
    std::vector<int> virt_host{};
    int nmo{0};
    int nph{0};
    int max_panel_width{0};
    std::size_t last_panel_width{0};
    DevicePqPhPanelStorageMode mode{DevicePqPhPanelStorageMode::StreamingPanel};
    DeviceBuffer<double> d_eri;
    DeviceBuffer<int> d_occ;
    DeviceBuffer<int> d_virt;
    DeviceBuffer<cuDoubleComplex> d_panel;
    PinnedHostBuffer<cuDoubleComplex> h_panel;
};

DevicePqPhPanelView::DevicePqPhPanelView(const MolecularIntegrals& integrals,
                                         const ParticleHoleBasis& ph_basis,
                                         std::size_t max_panel_width,
                                         DevicePqPhPanelStorageMode mode)
    : impl_(std::make_unique<Impl>(integrals, ph_basis, max_panel_width, mode)) {}

DevicePqPhPanelView::~DevicePqPhPanelView() = default;
DevicePqPhPanelView::DevicePqPhPanelView(DevicePqPhPanelView&&) noexcept = default;
DevicePqPhPanelView& DevicePqPhPanelView::operator=(DevicePqPhPanelView&&) noexcept = default;

std::size_t DevicePqPhPanelView::nmo() const noexcept {
    return impl_ ? static_cast<std::size_t>(impl_->nmo) : 0;
}

std::size_t DevicePqPhPanelView::nph() const noexcept {
    return impl_ ? static_cast<std::size_t>(impl_->nph) : 0;
}

std::size_t DevicePqPhPanelView::max_panel_width() const noexcept {
    return impl_ ? static_cast<std::size_t>(impl_->max_panel_width) : 0;
}

std::size_t DevicePqPhPanelView::last_panel_width() const noexcept {
    return impl_ ? impl_->last_panel_width : 0;
}

std::size_t DevicePqPhPanelView::estimated_device_bytes() const noexcept {
    return impl_ ? impl_->estimated_device_bytes() : 0;
}

std::size_t DevicePqPhPanelView::estimated_host_pinned_bytes() const noexcept {
    return impl_ ? impl_->estimated_host_pinned_bytes() : 0;
}

DevicePqPhPanelStorageMode DevicePqPhPanelView::storage_mode() const noexcept {
    return impl_ ? impl_->mode : DevicePqPhPanelStorageMode::StreamingPanel;
}

std::string_view DevicePqPhPanelView::storage_mode_name() const noexcept {
    return to_string(storage_mode());
}

DeviceComplexPanelView DevicePqPhPanelView::fill_panel(std::size_t p_index,
                                                       std::size_t k_begin,
                                                       std::size_t width) {
    if (!impl_) {
        throw std::runtime_error("DevicePqPhPanelView::fill_panel called on moved-from object");
    }
    return impl_->fill_panel(p_index, k_begin, width);
}

} // namespace gw::workspace
