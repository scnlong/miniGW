#include "gw/qp_solver.hpp"

#include "gw/pade.hpp"

#include <cmath>
#include <limits>

namespace gw {

std::vector<double> solve_qp_energies(const OrbitalSpace& orbitals,
                                      const MolecularIntegrals& integrals,
                                      const MatrixReal& sigma_x,
                                      const MatrixComplex& sigma_c_im_points,
                                      const std::vector<Complex>& omega_im,
                                      const std::vector<std::size_t>& states,
                                      const GwSettings& settings) {
    std::vector<double> qp_energy(orbitals.nmo(), std::numeric_limits<double>::quiet_NaN());
    constexpr std::size_t max_iterations = 200;
    constexpr double tolerance = 1e-6;

    for (const std::size_t state : states) {
        double current_qp = orbitals.energy(state);
        std::vector<Complex> orbital_sigma(settings.num_freq_points_total);
        for (std::size_t f = 0; f < settings.num_freq_points_total; ++f) {
            orbital_sigma[f] = sigma_c_im_points(state, f);
        }

        const auto [coeffs, sampled] =
            get_pade_coefficients_continued_fraction(omega_im, orbital_sigma, settings.num_pade_params);

        for (std::size_t iter = 0; iter < max_iterations; ++iter) {
            const double target_omega_qp = current_qp - orbitals.fermi_energy();
            const Complex sigma_c_at_qp = evaluate_pade_continued_fraction(coeffs, target_omega_qp, sampled);
            const double new_qp = orbitals.energy(state)
                                + sigma_c_at_qp.real()
                                + sigma_x(state, state)
                                - integrals.vxc(state, state);
            if (std::abs(new_qp - current_qp) < tolerance) {
                current_qp = new_qp;
                break;
            }
            current_qp = new_qp;
        }
        qp_energy[state] = current_qp;
    }
    return qp_energy;
}

} // namespace gw
