#include "gw/io.hpp"
#include "gw/pade.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace gw {

int read_int_text(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open integer text file: " + path);
    }
    int value = 0;
    in >> value;
    return value;
}

double read_double_text(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open double text file: " + path);
    }
    double value = 0.0;
    in >> value;
    return value;
}

void output_frequency_grid(const std::string& path, const std::vector<double>& omegas, const std::vector<double>& weights) {
    std::ofstream out(path);
    for (std::size_t i = 0; i < omegas.size(); ++i) {
        out << std::setprecision(17) << omegas[i] << ' ' << weights[i] << '\n';
    }
}

void output_self_energy_before_pade(const std::string& path, const MatrixComplex& sigma_c, std::size_t state_0based) {
    std::ofstream out(path);
    for (std::size_t i = 0; i < sigma_c.cols(); ++i) {
        const auto z = sigma_c(state_0based, i);
        out << '(' << std::setprecision(17) << z.real() << ',' << z.imag() << ")\n";
    }
}

void output_self_energy_after_pade(const std::string& path, const std::vector<Complex>& state, const std::vector<double>& omegas_re,
                                   std::size_t num_pade_params, std::size_t num_points, double fermi_energy,
                                   double x_start, double interval) {
    std::ofstream out(path);
    std::vector<Complex> omegas_im(omegas_re.size());
    for (std::size_t i = 0; i < omegas_re.size(); ++i) {
        omegas_im[i] = Complex{0.0, omegas_re[i]};
    }
    const auto [coeffs, sampled] = get_pade_coefficients_continued_fraction(omegas_im, state, num_pade_params);
    for (std::size_t i = 0; i < num_points; ++i) {
        const double e_iter = x_start + static_cast<double>(i) * interval - fermi_energy;
        const Complex e_c = evaluate_pade_continued_fraction(coeffs, e_iter, sampled);
        out << std::setprecision(17) << (e_iter + fermi_energy) * kHartreeToEv << ' ' << e_c.real() << ' ' << e_c.imag() << '\n';
    }
}

void output_final_energy_table(const std::vector<double>& mo_energy, const MatrixReal& sigma_x, const MatrixReal& vxc, const std::vector<double>& qp_energy,
                               const std::vector<std::size_t>& states_0based) {
    std::cout << std::left << std::setw(10) << "Orbital" << std::setw(15) << "KS Energy" << std::setw(15) << "Sigma_x"
              << std::setw(15) << "Sigma_c" << std::setw(15) << "Vxc" << std::setw(15) << "QP Energy" << '\n';
    std::cout << std::left << std::setw(10) << "-------" << std::setw(15) << "---------" << std::setw(15) << "-------"
              << std::setw(15) << "-------" << std::setw(15) << "-------" << std::setw(15) << "---------" << '\n';

    for (const auto idx : states_0based) {
        const double sigma_c_qp = qp_energy[idx] + vxc(idx, idx) - mo_energy[idx] - sigma_x(idx, idx);
        std::cout << std::left << std::setw(10) << (idx + 1)
                  << std::setw(15) << mo_energy[idx] * kHartreeToEv
                  << std::setw(15) << sigma_x(idx, idx) * kHartreeToEv
                  << std::setw(15) << sigma_c_qp * kHartreeToEv
                  << std::setw(15) << vxc(idx, idx) * kHartreeToEv
                  << std::setw(15) << qp_energy[idx] * kHartreeToEv << '\n';
    }
}

} // namespace gw
