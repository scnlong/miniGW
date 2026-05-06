#pragma once

#include "gw/orbital_space.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace gw {

struct ParticleHolePair {
    std::size_t i_occ{};  // global MO index, guaranteed occupied
    std::size_t a_vir{};  // local virtual index, 0 ... nvir-1
    std::size_t a_mo{};   // global MO index, guaranteed virtual
    double delta_e{};     // epsilon_a - epsilon_i
};

class ParticleHoleBasis {
public:
    ParticleHoleBasis() = default;

    explicit ParticleHoleBasis(const OrbitalSpace& orbitals) {
        pairs_.reserve(orbitals.nocc() * orbitals.nvir());
        for (std::size_t i_occ = 0; i_occ < orbitals.nocc(); ++i_occ) {
            for (std::size_t a_vir = 0; a_vir < orbitals.nvir(); ++a_vir) {
                const std::size_t a_mo = orbitals.virtual_to_mo(a_vir);
                pairs_.push_back(ParticleHolePair{
                    i_occ,
                    a_vir,
                    a_mo,
                    orbitals.energy(a_mo) - orbitals.energy(i_occ)
                });
            }
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return pairs_.size(); }

    [[nodiscard]] const ParticleHolePair& operator[](std::size_t ph_index) const {
#ifndef NDEBUG
        if (ph_index >= pairs_.size()) {
            throw std::out_of_range("ParticleHoleBasis: ph index out of range");
        }
#endif
        return pairs_[ph_index];
    }

    [[nodiscard]] std::size_t index(std::size_t i_occ, std::size_t a_vir, const OrbitalSpace& orbitals) const {
#ifndef NDEBUG
        if (i_occ >= orbitals.nocc() || a_vir >= orbitals.nvir()) {
            throw std::out_of_range("ParticleHoleBasis: invalid occupied/virtual pair");
        }
#endif
        return i_occ * orbitals.nvir() + a_vir;
    }

private:
    std::vector<ParticleHolePair> pairs_{};
};

} // namespace gw
