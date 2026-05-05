#pragma once

#include <cstddef>
#include <utility>

namespace gw {

inline std::size_t ph_to_idx(std::size_t i_occ, std::size_t a_mo, std::size_t nocc, std::size_t nvirt) {
    // C++ version uses 0-based indices. i_occ in [0, nocc), a_mo in [nocc, nocc+nvirt).
    return i_occ * nvirt + (a_mo - nocc);
}

inline std::pair<std::size_t, std::size_t> idx_to_ph(std::size_t idx, std::size_t nocc, std::size_t nvirt) {
    const std::size_t i_occ = idx / nvirt;
    const std::size_t a_mo = nocc + (idx % nvirt);
    return {i_occ, a_mo};
}

} // namespace gw
