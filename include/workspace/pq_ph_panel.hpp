#pragma once

#include "gw/integrals.hpp"
#include "gw/particle_hole.hpp"
#include "types.hpp"

#include <cstddef>

namespace gw::workspace {

void set_pq_ph_panel_openmp_kernel_loops(bool enabled) noexcept;

// Lightweight panel view for the pq-particle-hole coupling tensor
//   P_{pk,ia} = (p q | i a)
// without materializing the full Tensor3Real(nmo, nmo, n_ph).
//
// The current implementation still reads from replicated ERI stored in
// MolecularIntegrals.  Its purpose is to remove the resident pq_ph tensor and
// provide a single panel-generation boundary that can later be backed by
// distributed/tiled ERI storage.
class PqPhPanelView {
public:
    PqPhPanelView(const MolecularIntegrals& integrals,
                  const ParticleHoleBasis& ph_basis);

    [[nodiscard]] std::size_t nmo() const noexcept { return nmo_; }
    [[nodiscard]] std::size_t nph() const noexcept { return ph_basis_.size(); }

    // Return a dense host panel X with shape n_ph x width.  Column kk contains
    // pq_ph(p_index, k_begin + kk, :).  The caller chooses width as a small
    // contraction panel size; this avoids storing all nmo*nmo*n_ph entries.
    [[nodiscard]] MatrixReal make_panel(std::size_t p_index,
                                        std::size_t k_begin,
                                        std::size_t width) const;

    void fill_panel(std::size_t p_index,
                    std::size_t k_begin,
                    MatrixReal& panel) const;

private:
    const MolecularIntegrals& integrals_;
    const ParticleHoleBasis& ph_basis_;
    std::size_t nmo_{0};
};

[[nodiscard]] long double materialized_pq_ph_bytes(std::size_t nmo,
                                                   std::size_t nph);

} // namespace gw::workspace
