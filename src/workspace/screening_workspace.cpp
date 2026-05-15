#include "workspace/screening_workspace.hpp"

#include "runtime/kernel_policy.hpp"

#include <stdexcept>

namespace gw::workspace {

void set_host_screening_openmp_kernel_loops(bool enabled) noexcept {
    runtime::set_openmp_kernel_loops_enabled(enabled);
}

HostScreeningWorkspace::HostScreeningWorkspace(MatrixReal v_ph,
                                               MatrixComplex inv_v,
                                               const linalg::Backend& backend)
    : v_ph_(std::move(v_ph)), inv_v_(std::move(inv_v)), backend_(backend) {
    if (v_ph_.rows() != v_ph_.cols()) {
        throw std::runtime_error("HostScreeningWorkspace: V_ph must be square");
    }
    if (inv_v_.rows() != v_ph_.rows() || inv_v_.cols() != v_ph_.cols()) {
        throw std::runtime_error("HostScreeningWorkspace: inv(V_ph) shape is inconsistent with V_ph");
    }
}

MatrixComplex HostScreeningWorkspace::compute_w_c(const std::vector<Complex>& pi0_diag) const {
    const std::size_t n = v_ph_.rows();
    if (pi0_diag.size() != n) {
        throw std::runtime_error("HostScreeningWorkspace::compute_w_c: inconsistent pi0 dimension");
    }

    MatrixComplex epsilon(n, n, Complex{0.0, 0.0});
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for collapse(2) schedule(static) if(runtime::openmp_kernel_loops_enabled())
#endif
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            epsilon(i, k) = -Complex{v_ph_(i, k), 0.0} * pi0_diag[k];
        }
    }
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for schedule(static) if(runtime::openmp_kernel_loops_enabled())
#endif
    for (std::size_t i = 0; i < n; ++i) {
        epsilon(i, i) += Complex{1.0, 0.0};
    }

    MatrixComplex inv_eps = backend_.inverse(epsilon);
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for schedule(static) if(runtime::openmp_kernel_loops_enabled())
#endif
    for (std::size_t i = 0; i < n; ++i) {
        inv_eps(i, i) -= Complex{1.0, 0.0};
    }

    return backend_.gemm(inv_eps, inv_v_, linalg::MatrixTranspose::Transpose, linalg::MatrixTranspose::NoTranspose);
}


std::vector<Complex> HostScreeningWorkspace::quadratic_forms_panel(const MatrixComplex& w_c,
                                                                   const MatrixReal& x_panel) const {
    const std::size_t n = v_ph_.rows();
    if (w_c.rows() != n || w_c.cols() != n || x_panel.rows() != n) {
        throw std::runtime_error("HostScreeningWorkspace::quadratic_forms_panel: inconsistent dimensions");
    }

    const std::size_t nvec = x_panel.cols();
    MatrixComplex x_complex(n, nvec, Complex{0.0, 0.0});
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for collapse(2) schedule(static) if(runtime::openmp_kernel_loops_enabled())
#endif
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < nvec; ++j) {
            x_complex(i, j) = Complex{x_panel(i, j), 0.0};
        }
    }

    const MatrixComplex y = backend_.gemm(w_c, x_complex,
                                          linalg::MatrixTranspose::NoTranspose,
                                          linalg::MatrixTranspose::NoTranspose);
    std::vector<Complex> values(nvec, Complex{0.0, 0.0});
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for schedule(static) if(runtime::openmp_kernel_loops_enabled())
#endif
    for (std::size_t j = 0; j < nvec; ++j) {
        Complex sum{0.0, 0.0};
        for (std::size_t i = 0; i < n; ++i) {
            sum += x_panel(i, j) * y(i, j);
        }
        values[j] = sum;
    }
    return values;
}

} // namespace gw::workspace
