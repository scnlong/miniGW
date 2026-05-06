#include "gw/memory_footprint.hpp"

#include "gw/types.hpp"

#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace gw {
namespace {

[[nodiscard]] long double bytes_for(std::initializer_list<std::size_t> dims,
                                    std::size_t element_size) {
    long double bytes = static_cast<long double>(element_size);
    for (const std::size_t dim : dims) {
        bytes *= static_cast<long double>(dim);
    }
    return bytes;
}

[[nodiscard]] std::string format_bytes(long double bytes) {
    constexpr long double kib = 1024.0L;
    constexpr long double mib = kib * 1024.0L;
    constexpr long double gib = mib * 1024.0L;
    constexpr long double tib = gib * 1024.0L;

    std::ostringstream out;
    out << std::fixed << std::setprecision(1);
    if (bytes >= tib) {
        out << static_cast<double>(bytes / tib) << " TB";
    } else if (bytes >= gib) {
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

long double MemoryFootprint::input_bytes() const {
    return input_eri_bytes + input_vxc_bytes + input_mo_energy_bytes;
}

long double MemoryFootprint::runtime_copy_bytes() const {
    return integrals_eri_copy_bytes + integrals_vxc_copy_bytes;
}

long double MemoryFootprint::persistent_runtime_bytes_per_rank() const {
    return input_bytes()
         + runtime_copy_bytes()
         + sigma_x_bytes
         + v_ph_matrix_bytes
         + pq_ph_tensor_bytes
         + sigma_c_bytes
         + qp_energy_bytes
         + frequency_grid_bytes
         + inv_v_bytes;
}

long double MemoryFootprint::per_frequency_workspace_bytes_one_replica() const {
    return pi0_diag_bytes
         + pk_vec_bytes
         + current_sigma_bytes
         + quadratic_form_workspace_bytes
         + w_c_construction_peak_bytes;
}

long double MemoryFootprint::replicated_workspace_bytes_per_rank() const {
    return static_cast<long double>(frequency_workspace_replicas) * per_frequency_workspace_bytes_one_replica();
}

long double MemoryFootprint::peak_bytes_per_rank() const {
    return persistent_runtime_bytes_per_rank() + replicated_workspace_bytes_per_rank();
}

long double MemoryFootprint::aggregate_peak_bytes_all_ranks() const {
    return static_cast<long double>(mpi_size) * peak_bytes_per_rank();
}

MemoryFootprint estimate_memory_footprint(const GwInput& input,
                                          const GwSettings& settings) {
    const std::size_t nmo = input.mo_energy.size();
    if (input.nocc > nmo) {
        throw std::runtime_error("estimate_memory_footprint: nocc exceeds number of molecular orbitals");
    }

    const std::size_t nvirt = nmo - input.nocc;
    const std::size_t n_ph = input.nocc * nvirt;

    MemoryFootprint footprint;
    footprint.nmo = nmo;
    footprint.nocc = input.nocc;
    footprint.nvirt = nvirt;
    footprint.n_ph = n_ph;
    footprint.num_freq_points = settings.num_freq_points_total;
    footprint.mpi_enabled = settings.execution.mpi_size > 1;
    footprint.mpi_size = settings.execution.mpi_size;
    footprint.frequency_parallel_mode = settings.execution.frequency_parallel_mode;
    footprint.frequency_workspace_replicas = settings.execution.frequency_workspace_replicas;

    footprint.input_eri_bytes = bytes_for({input.eri_mo.dim0(),
                                           input.eri_mo.dim1(),
                                           input.eri_mo.dim2(),
                                           input.eri_mo.dim3()},
                                          sizeof(double));
    footprint.input_vxc_bytes = bytes_for({input.vxc_mo.rows(), input.vxc_mo.cols()},
                                          sizeof(double));
    footprint.input_mo_energy_bytes = bytes_for({nmo}, sizeof(double));

    // Current code constructs MolecularIntegrals(input.eri_mo, input.vxc_mo)
    // and MolecularIntegrals owns its Tensor4Real/MatrixReal by value. Count
    // these resident copies explicitly; they dominate for realistic nmo.
    footprint.integrals_eri_copy_bytes = footprint.input_eri_bytes;
    footprint.integrals_vxc_copy_bytes = footprint.input_vxc_bytes;

    footprint.sigma_x_bytes = bytes_for({nmo, nmo}, sizeof(double));
    footprint.v_ph_matrix_bytes = bytes_for({n_ph, n_ph}, sizeof(double));
    footprint.pq_ph_tensor_bytes = bytes_for({nmo, nmo, n_ph}, sizeof(double));
    footprint.sigma_c_bytes = bytes_for({nmo, settings.num_freq_points_total}, sizeof(Complex));
    footprint.qp_energy_bytes = bytes_for({nmo}, sizeof(double));
    // gauss_legendre_grid gives omegas and weights; run_g0w0 also builds omega_im.
    footprint.frequency_grid_bytes = bytes_for({settings.num_freq_points_total}, sizeof(double))
                                   + bytes_for({settings.num_freq_points_total}, sizeof(double))
                                   + bytes_for({settings.num_freq_points_total}, sizeof(Complex));

    // inv(V_ph) is now computed once outside the double frequency loop and kept resident.
    footprint.inv_v_bytes = bytes_for({n_ph, n_ph}, sizeof(Complex));

    footprint.pi0_diag_bytes = bytes_for({n_ph}, sizeof(Complex));
    footprint.pk_vec_bytes = bytes_for({n_ph}, sizeof(double));
    footprint.current_sigma_bytes = bytes_for({nmo}, sizeof(Complex));
    footprint.quadratic_form_workspace_bytes = bytes_for({n_ph}, sizeof(Complex));

    footprint.w_c_matrix_bytes = bytes_for({n_ph, n_ph}, sizeof(Complex));

    // Conservative host-side peak estimate for the current inverse-based
    // implementation: epsilon, inv_eps, inv_v, returned W_c, an inverse()
    // argument copy, and inverse() internal work/identity storage. Optimized
    // solve-based backends can reduce this term; backend-internal workspaces are
    // not included here.
    footprint.w_c_construction_peak_bytes = 6.0L * footprint.w_c_matrix_bytes;

    return footprint;
}

MemoryFootprint estimate_memory_footprint(const GwInput& input,
                                          std::size_t num_freq_points) {
    GwSettings settings;
    settings.num_freq_points_total = num_freq_points;
    return estimate_memory_footprint(input, settings);
}

void print_memory_footprint_report(const MemoryFootprint& footprint) {
    std::cout << "\n--- Estimated host-side memory footprint --- \n";
    std::cout << "  Dimensions:\n";
    std::cout << "    frequency points:            " << footprint.num_freq_points << '\n';
    std::cout << "    frequency parallel mode:      " << to_string(footprint.frequency_parallel_mode) << '\n';
    std::cout << "    MPI ranks:                    " << footprint.mpi_size << '\n';
    std::cout << "    workspace replicas per rank:  " << footprint.frequency_workspace_replicas << '\n';

    std::cout << "  Input arrays resident in each rank's GwInput:\n";
    std::cout << "    ERI tensor:                   " << format_bytes(footprint.input_eri_bytes) << '\n';
    std::cout << "    Vxc MO matrix:                " << format_bytes(footprint.input_vxc_bytes) << '\n';
    std::cout << "    MO energies:                  " << format_bytes(footprint.input_mo_energy_bytes) << '\n';
    std::cout << "    Input subtotal:               " << format_bytes(footprint.input_bytes()) << '\n';

    std::cout << "  Current runtime resident copies per rank:\n";
    std::cout << "    ERI copy in integrals:        " << format_bytes(footprint.integrals_eri_copy_bytes) << '\n';
    std::cout << "    Vxc copy in integrals:        " << format_bytes(footprint.integrals_vxc_copy_bytes) << '\n';
    std::cout << "    Copy subtotal:                " << format_bytes(footprint.runtime_copy_bytes()) << '\n';

    std::cout << "  Main GW resident arrays per rank:\n";
    std::cout << "    Sigma_x matrix:               " << format_bytes(footprint.sigma_x_bytes) << '\n';
    std::cout << "    V_ph matrix:                  " << format_bytes(footprint.v_ph_matrix_bytes) << '\n';
    std::cout << "    inv(V_ph) matrix:             " << format_bytes(footprint.inv_v_bytes) << '\n';
    std::cout << "    pq_ph tensor:                 " << format_bytes(footprint.pq_ph_tensor_bytes) << '\n';
    std::cout << "    sigma_c(iw) grid:             " << format_bytes(footprint.sigma_c_bytes) << '\n';
    std::cout << "    QP energy vector:             " << format_bytes(footprint.qp_energy_bytes) << '\n';
    std::cout << "    frequency grids:              " << format_bytes(footprint.frequency_grid_bytes) << '\n';

    std::cout << "  Per-frequency workspace, one replica:\n";
    std::cout << "    pi0 diagonal:                 " << format_bytes(footprint.pi0_diag_bytes) << '\n';
    std::cout << "    pk vector:                    " << format_bytes(footprint.pk_vec_bytes) << '\n';
    std::cout << "    current sigma vector:         " << format_bytes(footprint.current_sigma_bytes) << '\n';
    std::cout << "    quadratic workspace:          " << format_bytes(footprint.quadratic_form_workspace_bytes) << '\n';
    std::cout << "    one W_c matrix:               " << format_bytes(footprint.w_c_matrix_bytes) << '\n';
    std::cout << "    W_c construction peak:        " << format_bytes(footprint.w_c_construction_peak_bytes) << '\n';

    std::cout << "  Totals:\n";
    std::cout << "    Persistent subtotal per rank: " << format_bytes(footprint.persistent_runtime_bytes_per_rank()) << '\n';
    std::cout << "    Workspace subtotal per rank:  " << format_bytes(footprint.replicated_workspace_bytes_per_rank()) << '\n';
    std::cout << "    Estimated peak per rank:      " << format_bytes(footprint.peak_bytes_per_rank()) << '\n';
    std::cout << "    Aggregate across all ranks:   " << format_bytes(footprint.aggregate_peak_bytes_all_ranks()) << '\n';
    std::cout << "  Note: the first MPI implementation replicates ERI, V_ph, inv(V_ph), pq_ph, and sigma_c on each rank.\n"
              << "        MPI frequency parallelism improves time-to-solution but does not reduce per-rank memory.\n"
              << "        This estimate excludes allocator overhead, BLAS/LAPACK/cuSolver internal workspaces, GPU memory, and OS/runtime overhead.\n\n";
}

} // namespace gw
