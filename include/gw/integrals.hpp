#pragma once

#include "types.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace gw {

class MolecularIntegrals {
public:
    MolecularIntegrals() = default;

    MolecularIntegrals(Tensor4Real eri_mo, MatrixReal vxc_mo)
        : eri_mo_(std::move(eri_mo)), vxc_mo_(std::move(vxc_mo)) {
        const std::size_t nmo = eri_mo_.dim0();
        if (nmo == 0 || eri_mo_.dim1() != nmo || eri_mo_.dim2() != nmo || eri_mo_.dim3() != nmo) {
            throw std::runtime_error("MolecularIntegrals: ERI tensor must have shape [nmo,nmo,nmo,nmo]");
        }
        if (vxc_mo_.rows() != nmo || vxc_mo_.cols() != nmo) {
            throw std::runtime_error("MolecularIntegrals: vxc_mo shape does not match ERI tensor");
        }
    }

    [[nodiscard]] std::size_t nmo() const noexcept { return eri_mo_.dim0(); }

    [[nodiscard]] double eri(std::size_t p, std::size_t q, std::size_t r, std::size_t s) const {
        // This returns the exact C-order payload read from /eri_mo in the HDF5 input file.
        // In this prototype it follows the same index order as the original Julia/NPZ implementation.
        return eri_mo_(p, q, r, s);
    }

    [[nodiscard]] double vxc(std::size_t p, std::size_t q) const {
        return vxc_mo_(p, q);
    }

    [[nodiscard]] const Tensor4Real& eri_tensor() const noexcept { return eri_mo_; }
    [[nodiscard]] const MatrixReal& vxc_matrix() const noexcept { return vxc_mo_; }

private:
    Tensor4Real eri_mo_{};
    MatrixReal vxc_mo_{};
};

} // namespace gw
