#pragma once

#include "gw/types.hpp"

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace gw {

class CoutTee {
public:
    explicit CoutTee(std::ostream& log_stream);
    ~CoutTee();

    CoutTee(const CoutTee&) = delete;
    CoutTee& operator=(const CoutTee&) = delete;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

int read_int_text(const std::string& path);
double read_double_text(const std::string& path);
void output_frequency_grid(const std::string& path, const std::vector<double>& omegas, const std::vector<double>& weights);
void output_self_energy_before_pade(const std::string& path, const MatrixComplex& sigma_c, std::size_t state_0based);
void output_self_energy_after_pade(const std::string& path, const std::vector<Complex>& state, const std::vector<double>& omegas_re,
                                   std::size_t num_pade_params, std::size_t num_points, double fermi_energy,
                                   double x_start = -6.5, double interval = 0.00065);
void output_final_energy_table(const std::vector<double>& mo_energy, const MatrixReal& sigma_x, const MatrixReal& vxc, const std::vector<double>& qp_energy,
                               const std::vector<std::size_t>& states_0based);

} // namespace gw
