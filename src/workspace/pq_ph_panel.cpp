#include "workspace/pq_ph_panel.hpp"

#include <atomic>
#include <stdexcept>

namespace gw::workspace {
namespace {

#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
std::atomic_bool g_pq_ph_panel_openmp_kernel_loops_enabled{true};

bool kernel_loops_enabled() noexcept {
    return g_pq_ph_panel_openmp_kernel_loops_enabled.load(std::memory_order_relaxed);
}
#else
bool kernel_loops_enabled() noexcept {
    return false;
}
#endif

} // namespace

void set_pq_ph_panel_openmp_kernel_loops(bool enabled) noexcept {
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
    g_pq_ph_panel_openmp_kernel_loops_enabled.store(enabled, std::memory_order_relaxed);
#else
    (void)enabled;
#endif
}

PqPhPanelView::PqPhPanelView(const MolecularIntegrals& integrals,
                             const ParticleHoleBasis& ph_basis)
    : integrals_(integrals), ph_basis_(ph_basis), nmo_(integrals.nmo()) {}

MatrixReal PqPhPanelView::make_panel(std::size_t p_index,
                                     std::size_t k_begin,
                                     std::size_t width) const {
    MatrixReal panel(nph(), width, 0.0);
    fill_panel(p_index, k_begin, panel);
    return panel;
}

void PqPhPanelView::fill_panel(std::size_t p_index,
                               std::size_t k_begin,
                               MatrixReal& panel) const {
    if (p_index >= nmo_) {
        throw std::runtime_error("PqPhPanelView::fill_panel: p index out of range");
    }
    if (panel.rows() != nph()) {
        throw std::runtime_error("PqPhPanelView::fill_panel: panel row count must equal n_ph");
    }
    if (k_begin > nmo_ || panel.cols() > nmo_ - k_begin) {
        throw std::runtime_error("PqPhPanelView::fill_panel: k panel exceeds nmo");
    }

#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for collapse(2) schedule(static) if(kernel_loops_enabled())
#endif
    for (std::size_t ph = 0; ph < ph_basis_.size(); ++ph) {
        for (std::size_t kk = 0; kk < panel.cols(); ++kk) {
            const ParticleHolePair& ia = ph_basis_[ph];
            panel(ph, kk) = integrals_.eri(p_index, k_begin + kk, ia.i_occ, ia.a_mo);
        }
    }
}

long double materialized_pq_ph_bytes(std::size_t nmo, std::size_t nph) {
    return static_cast<long double>(nmo) *
           static_cast<long double>(nmo) *
           static_cast<long double>(nph) *
           static_cast<long double>(sizeof(double));
}

} // namespace gw::workspace
