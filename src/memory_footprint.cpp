#include "gw/memory_footprint.hpp"

#include "gw/types.hpp"

#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>

namespace gw {
namespace {

long double bytes_for(std::initializer_list<std::size_t> dims, std::size_t element_size) {
    long double bytes = static_cast<long double>(element_size);
    for (const std::size_t dim : dims) {
        bytes *= static_cast<long double>(dim);
    }
    return bytes;
}

std::string format_bytes(long double bytes) {
    constexpr long double kib = 1024.0L;
    constexpr long double mib = kib * 1024.0L;
    constexpr long double gib = mib * 1024.0L;

    std::ostringstream out;
    out << std::fixed << std::setprecision(1);
    if (bytes >= gib) {
        out << static_cast<double>(bytes / gib) << " GB";
    } else if (bytes >= mib) {
        out << static_cast<double>(bytes / mib) << " MB";
    } else if (bytes >= kib) {
        out << static_cast<double>(bytes / kib) << " KB";
    } else {
        out << static_cast<double>(bytes) << " B";
    }
    return out.str();
}

} // namespace

long double MemoryFootprint::persistent_bytes() const {
    return eri_tensor_bytes + mo_energy_bytes + vxc_matrix_bytes + sigma_x_bytes
         + v_ph_matrix_bytes + pq_ph_tensor_bytes + sigma_c_bytes + frequency_grid_bytes;
}

long double MemoryFootprint::peak_bytes() const {
    return persistent_bytes() + pi0_diag_bytes + w_c_per_freq_bytes + w_c_workspace_bytes;
}

MemoryFootprint estimate_memory_footprint(const GwInput& input, std::size_t num_freq_points) {
    const std::size_t nmo = input.mo_energy.size();
    const std::size_t nvirt = nmo - input.nocc;
    const std::size_t n_ph = input.nocc * nvirt;

    MemoryFootprint footprint;
    footprint.eri_tensor_bytes = bytes_for({input.eri_mo.dim0(), input.eri_mo.dim1(), input.eri_mo.dim2(), input.eri_mo.dim3()}, sizeof(double));
    footprint.mo_energy_bytes = bytes_for({nmo}, sizeof(double));
    footprint.vxc_matrix_bytes = bytes_for({input.vxc_mo.rows(), input.vxc_mo.cols()}, sizeof(double));
    footprint.sigma_x_bytes = bytes_for({nmo, nmo}, sizeof(double));
    footprint.v_ph_matrix_bytes = bytes_for({n_ph, n_ph}, sizeof(double));
    footprint.pq_ph_tensor_bytes = bytes_for({nmo, nmo, n_ph}, sizeof(double));
    footprint.sigma_c_bytes = bytes_for({nmo, num_freq_points}, sizeof(Complex));
    footprint.frequency_grid_bytes = bytes_for({num_freq_points}, sizeof(double) * 2)   // omegas + weights
                                    + bytes_for({num_freq_points}, sizeof(Complex)); // omega_im
    footprint.pi0_diag_bytes = bytes_for({n_ph}, sizeof(Complex));
    footprint.w_c_per_freq_bytes = bytes_for({n_ph, n_ph}, sizeof(Complex));
    // During W_c construction, epsilon, inv_eps, inv_v, and the returned W_c
    // can coexist. Count the three extra complex matrices beyond W_c itself.
    footprint.w_c_workspace_bytes = 3.0L * bytes_for({n_ph, n_ph}, sizeof(Complex));
    return footprint;
}

void print_memory_footprint_report(const MemoryFootprint& footprint) {
    std::cout << "\nEstimated memory footprint: " << format_bytes(footprint.peak_bytes()) << '\n';
    std::cout << "  - ERI tensor:             " << format_bytes(footprint.eri_tensor_bytes) << '\n';
    std::cout << "  - MO energies:            " << format_bytes(footprint.mo_energy_bytes) << '\n';
    std::cout << "  - Vxc MO matrix:          " << format_bytes(footprint.vxc_matrix_bytes) << '\n';
    std::cout << "  - Sigma_x matrix:         " << format_bytes(footprint.sigma_x_bytes) << '\n';
    std::cout << "  - V_ph matrix:            " << format_bytes(footprint.v_ph_matrix_bytes) << '\n';
    std::cout << "  - pq_ph tensor:           " << format_bytes(footprint.pq_ph_tensor_bytes) << '\n';
    std::cout << "  - sigma_c grid:           " << format_bytes(footprint.sigma_c_bytes) << '\n';
    std::cout << "  - frequency grids:        " << format_bytes(footprint.frequency_grid_bytes) << '\n';
    std::cout << "  - pi0 diag per freq:      " << format_bytes(footprint.pi0_diag_bytes) << '\n';
    std::cout << "  - W_c per freq:           " << format_bytes(footprint.w_c_per_freq_bytes) << '\n';
    std::cout << "  - W_c temporaries:        " << format_bytes(footprint.w_c_workspace_bytes) << '\n';
    std::cout << "  - Persistent subtotal:    " << format_bytes(footprint.persistent_bytes()) << '\n';
    std::cout << "  - Peak working estimate:  " << format_bytes(footprint.peak_bytes()) << "\n\n";
}

} // namespace gw
