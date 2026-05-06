#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gw {

class OrbitalSpace {
public:
    OrbitalSpace() = default;

    OrbitalSpace(std::vector<double> mo_energy, std::size_t nocc, double fermi_energy)
        : mo_energy_(std::move(mo_energy)), nocc_(nocc), fermi_energy_(fermi_energy) {
        if (mo_energy_.empty()) {
            throw std::runtime_error("OrbitalSpace: empty MO energy vector");
        }
        if (nocc_ == 0 || nocc_ >= mo_energy_.size()) {
            throw std::runtime_error("OrbitalSpace: invalid number of occupied orbitals");
        }
    }

    [[nodiscard]] std::size_t nmo() const noexcept { return mo_energy_.size(); }
    [[nodiscard]] std::size_t nocc() const noexcept { return nocc_; }
    [[nodiscard]] std::size_t nvir() const noexcept { return nmo() - nocc_; }
    [[nodiscard]] double fermi_energy() const noexcept { return fermi_energy_; }

    [[nodiscard]] const std::vector<double>& energies() const noexcept { return mo_energy_; }

    [[nodiscard]] double energy(std::size_t mo_index) const {
#ifndef NDEBUG
        if (mo_index >= nmo()) {
            throw std::out_of_range("OrbitalSpace: MO index out of range");
        }
#endif
        return mo_energy_[mo_index];
    }

    [[nodiscard]] bool is_occupied(std::size_t mo_index) const noexcept {
        return mo_index < nocc_;
    }

    [[nodiscard]] bool is_virtual(std::size_t mo_index) const noexcept {
        return mo_index >= nocc_ && mo_index < nmo();
    }

    [[nodiscard]] std::size_t virtual_to_mo(std::size_t virtual_index) const {
#ifndef NDEBUG
        if (virtual_index >= nvir()) {
            throw std::out_of_range("OrbitalSpace: virtual index out of range");
        }
#endif
        return nocc_ + virtual_index;
    }

    [[nodiscard]] std::size_t mo_to_virtual(std::size_t mo_index) const {
#ifndef NDEBUG
        if (!is_virtual(mo_index)) {
            throw std::out_of_range("OrbitalSpace: MO index is not virtual");
        }
#endif
        return mo_index - nocc_;
    }

private:
    std::vector<double> mo_energy_{};
    std::size_t nocc_{0};
    double fermi_energy_{0.0};
};

} // namespace gw
