#include "memory_footprint.hpp"

#include "types.hpp"

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
         + frequency_grid_bytes;
}

long double MemoryFootprint::per_frequency_workspace_bytes_one_replica() const {
    return pi0_diag_bytes
         + pk_vec_bytes
         + pk_panel_bytes
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

long double MemoryFootprint::estimated_device_bytes_per_rank() const {
    return device_v_ph_bytes
         + device_epsilon_bytes
         + device_w_c_bytes
         + device_pi0_bytes
         + device_solver_workspace_bytes
         + device_panel_workspace_bytes
         + device_eri_bytes
         + device_ph_index_bytes
         + device_pq_panel_bytes;
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
    footprint.pq_ph_tensor_bytes = 0.0L;
    footprint.avoided_materialized_pq_ph_bytes = bytes_for({nmo, nmo, n_ph}, sizeof(double));
    footprint.sigma_c_bytes = bytes_for({nmo, settings.num_freq_points_total}, sizeof(Complex));
    footprint.qp_energy_bytes = bytes_for({nmo}, sizeof(double));
    // gauss_legendre_grid gives omegas and weights; run_g0w0 also builds omega_im.
    footprint.frequency_grid_bytes = bytes_for({settings.num_freq_points_total}, sizeof(double))
                                   + bytes_for({settings.num_freq_points_total}, sizeof(double))
                                   + bytes_for({settings.num_freq_points_total}, sizeof(Complex));

    footprint.pi0_diag_bytes = bytes_for({n_ph}, sizeof(Complex));
    footprint.pk_vec_bytes = bytes_for({n_ph}, sizeof(double));
    footprint.pk_panel_bytes = bytes_for({n_ph, settings.contraction_panel_size}, sizeof(double));
    footprint.current_sigma_bytes = bytes_for({nmo}, sizeof(Complex));
    footprint.quadratic_form_workspace_bytes = bytes_for({n_ph}, sizeof(Complex));

    footprint.w_c_matrix_bytes = bytes_for({n_ph, n_ph}, sizeof(Complex));

    // Conservative host-side peak estimate for the no-inv(V_ph) screening
    // construction: the left dielectric matrix, W_c / solve right-hand side,
    // and inverse/solve internal work or identity storage. Backend-internal
    // workspaces are not included here.
    footprint.w_c_construction_peak_bytes = 4.0L * footprint.w_c_matrix_bytes;
    if (settings.linalg_backend && settings.linalg_backend->capabilities().uses_device_memory) {
        // DeviceScreeningWorkspace stores V_ph as complex column-major data,
        // the left dielectric matrix I - diag(Pi0) V_ph, and W_c.  It no longer
        // keeps inv(V_ph) or inv(epsilon)-I resident.  cuSolver workspace is
        // queried at runtime, so use one n_ph^2 complex matrix as a conservative
        // planning estimate here.  The panel term covers X, Y, and the small
        // output vector for panel quadratic forms.
        footprint.device_v_ph_bytes = footprint.w_c_matrix_bytes;
        footprint.device_epsilon_bytes = footprint.w_c_matrix_bytes;
        footprint.device_w_c_bytes = footprint.w_c_matrix_bytes;
        footprint.device_pi0_bytes = footprint.pi0_diag_bytes;
        footprint.device_solver_workspace_bytes = footprint.w_c_matrix_bytes
                                                + bytes_for({n_ph}, sizeof(int))
                                                + bytes_for({1}, sizeof(int));
        footprint.device_panel_workspace_bytes = bytes_for({n_ph, settings.contraction_panel_size}, sizeof(Complex))
                                                + bytes_for({settings.contraction_panel_size}, sizeof(Complex));
        // DevicePqPhPanelView now auto-selects between a full device ERI copy and
        // a pinned-host streaming panel source.  Static memory reporting cannot
        // know the runtime device-memory decision, so report the safer
        // streaming mode as the planning baseline and separately show the full
        // ERI copy that would be needed by the full-resident mode.
        footprint.device_eri_bytes = 0.0L;
        footprint.device_eri_streaming_saved_bytes = footprint.input_eri_bytes;
        footprint.device_ph_index_bytes = 2.0L * bytes_for({n_ph}, sizeof(int));
        footprint.device_pq_panel_bytes = bytes_for({n_ph, settings.contraction_panel_size}, sizeof(Complex));
        footprint.device_pq_panel_pinned_host_bytes = footprint.device_pq_panel_bytes;
    }

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
    std::cout << "    pq_ph tensor resident:        " << format_bytes(footprint.pq_ph_tensor_bytes) << '\n';
    std::cout << "    pq_ph tensor avoided:         " << format_bytes(footprint.avoided_materialized_pq_ph_bytes) << '\n';
    std::cout << "    sigma_c(iw) grid:             " << format_bytes(footprint.sigma_c_bytes) << '\n';
    std::cout << "    QP energy vector:             " << format_bytes(footprint.qp_energy_bytes) << '\n';
    std::cout << "    frequency grids:              " << format_bytes(footprint.frequency_grid_bytes) << '\n';

    std::cout << "  Per-frequency workspace, one replica:\n";
    std::cout << "    pi0 diagonal:                 " << format_bytes(footprint.pi0_diag_bytes) << '\n';
    std::cout << "    pk vector (legacy scalar):    " << format_bytes(footprint.pk_vec_bytes) << '\n';
    std::cout << "    pk panel:                     " << format_bytes(footprint.pk_panel_bytes) << '\n';
    std::cout << "    current sigma vector:         " << format_bytes(footprint.current_sigma_bytes) << '\n';
    std::cout << "    quadratic workspace:          " << format_bytes(footprint.quadratic_form_workspace_bytes) << '\n';
    std::cout << "    one W_c matrix:               " << format_bytes(footprint.w_c_matrix_bytes) << '\n';
    std::cout << "    W_c construction peak:        " << format_bytes(footprint.w_c_construction_peak_bytes) << '\n';

    if (footprint.estimated_device_bytes_per_rank() > 0.0L) {
        std::cout << "  Estimated GPU device memory per rank:\n";
        std::cout << "    V_ph(device):                 " << format_bytes(footprint.device_v_ph_bytes) << '\n';
		std::cout << "    epsilon_left(device):         " << format_bytes(footprint.device_epsilon_bytes) << '\n';
        std::cout << "    W_c(device):                  " << format_bytes(footprint.device_w_c_bytes) << '\n';
        std::cout << "    pi0(device):                  " << format_bytes(footprint.device_pi0_bytes) << '\n';
        std::cout << "    solver workspace estimate:    " << format_bytes(footprint.device_solver_workspace_bytes) << '\n';
        std::cout << "    contraction Y/q workspace:    " << format_bytes(footprint.device_panel_workspace_bytes) << '\n';
        std::cout << "    ERI(device, if resident mode): " << format_bytes(footprint.device_eri_bytes) << '\n';
        std::cout << "    full ERI device copy avoided: " << format_bytes(footprint.device_eri_streaming_saved_bytes) << '\n';
        std::cout << "    ph index arrays(device):       " << format_bytes(footprint.device_ph_index_bytes) << '\n';
        std::cout << "    pq X panel(device):            " << format_bytes(footprint.device_pq_panel_bytes) << '\n';
        std::cout << "    pq X pinned host staging:      " << format_bytes(footprint.device_pq_panel_pinned_host_bytes) << '\n';
        std::cout << "    Estimated device subtotal:    " << format_bytes(footprint.estimated_device_bytes_per_rank()) << '\n';
    }

    std::cout << "  Totals:\n";
    std::cout << "    Persistent subtotal per rank: " << format_bytes(footprint.persistent_runtime_bytes_per_rank()) << '\n';
    std::cout << "    Workspace subtotal per rank:  " << format_bytes(footprint.replicated_workspace_bytes_per_rank()) << '\n';
    std::cout << "    Estimated peak per rank:      " << format_bytes(footprint.peak_bytes_per_rank()) << '\n';
    std::cout << "    Aggregate across all ranks:   " << format_bytes(footprint.aggregate_peak_bytes_all_ranks()) << '\n';
    std::cout << "  Note: this version no longer materializes the full pq_ph tensor; pk panels are generated on demand.\n"
		      << "        ERI, V_ph, and sigma_c may still be replicated depending on the selected backend.\n"
              << "        MPI frequency parallelism improves time-to-solution but does not reduce per-rank memory.\n"
              << "        The self-energy contraction batches (p,k) vectors into panels; this estimate includes one such panel per workspace replica.\n"
              << "        Host totals exclude allocator overhead, BLAS/LAPACK workspaces, and OS/runtime overhead.\n"
              << "        GPU device totals are planning estimates; the actual cuSolver workspace is queried at runtime.\n\n";
}

} // namespace gw
