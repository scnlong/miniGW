#include "gw/sigma.hpp"

#include "runtime/kernel_policy.hpp"

#include <stdexcept>

namespace gw {
namespace {

void validate_input_shapes(const OrbitalSpace& orbitals, const MolecularIntegrals& integrals) {
    if (integrals.nmo() != orbitals.nmo()) {
        throw std::runtime_error("Input shape mismatch: integrals.nmo() != orbitals.nmo()");
    }
}

} // namespace

void set_sigma_openmp_kernel_loops(bool enabled) noexcept {
    runtime::set_openmp_kernel_loops_enabled(enabled);
}

MatrixReal calculate_exchange(const OrbitalSpace& orbitals, const MolecularIntegrals& integrals) {
    validate_input_shapes(orbitals, integrals);

    MatrixReal sigma_x(orbitals.nmo(), orbitals.nmo(), 0.0);
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for collapse(2) schedule(static) if(runtime::openmp_kernel_loops_enabled())
#endif
    for (std::size_t q = 0; q < orbitals.nmo(); ++q) {
        for (std::size_t p = 0; p < orbitals.nmo(); ++p) {
            double value = 0.0;
            for (std::size_t k_occ = 0; k_occ < orbitals.nocc(); ++k_occ) {
                value -= integrals.eri(p, k_occ, q, k_occ);
            }
            sigma_x(p, q) = value;
        }
    }
    return sigma_x;
}

std::vector<Complex> calculate_pi0_ph_diag(Complex omega, const ParticleHoleBasis& ph_basis, double eta) {
    std::vector<Complex> diag(ph_basis.size());
    const Complex ieta{0.0, eta};
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for schedule(static) if(runtime::openmp_kernel_loops_enabled())
#endif
    for (std::size_t ph = 0; ph < ph_basis.size(); ++ph) {
        const ParticleHolePair& ia = ph_basis[ph];
        const Complex term = Complex{1.0, 0.0} / (omega - ia.delta_e + ieta)
                           - Complex{1.0, 0.0} / (omega + ia.delta_e - ieta);
        diag[ph] = 2.0 * term;
    }
    return diag;
}

MatrixReal calculate_v_ph_matrix(const MolecularIntegrals& integrals, const ParticleHoleBasis& ph_basis) {
    MatrixReal v_ph(ph_basis.size(), ph_basis.size(), 0.0);
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for collapse(2) schedule(static) if(runtime::openmp_kernel_loops_enabled())
#endif
    for (std::size_t ph_ia = 0; ph_ia < ph_basis.size(); ++ph_ia) {
        for (std::size_t ph_jb = 0; ph_jb < ph_basis.size(); ++ph_jb) {
            const ParticleHolePair& ia = ph_basis[ph_ia];
            const ParticleHolePair& jb = ph_basis[ph_jb];
            // Convention used by epsilon: epsilon[row, col] = delta[row, col] - V_ph[row, col] * Pi0[col].
            // Therefore row = jb, col = ia, and V_ph[jb, ia] = (j b | i a).
            v_ph(ph_jb, ph_ia) = integrals.eri(jb.i_occ, jb.a_mo, ia.i_occ, ia.a_mo);
        }
    }
    return v_ph;
}

MatrixComplex calculate_w_0_c_matrix(const MatrixReal& v_ph,
                                     const MatrixComplex& inv_v,
                                     const std::vector<Complex>& pi0_diag,
                                     const linalg::Backend& backend) {
    const std::size_t n = v_ph.rows();
    if (v_ph.rows() != v_ph.cols() || pi0_diag.size() != n || inv_v.rows() != n || inv_v.cols() != n) {
        throw std::runtime_error("calculate_w_0_c_matrix: inconsistent dimensions");
    }

    MatrixComplex epsilon(n, n, Complex{0.0, 0.0});
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for collapse(2) schedule(static) if(runtime::openmp_kernel_loops_enabled())
#endif
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            epsilon(i, k) = -Complex{v_ph(i, k), 0.0} * pi0_diag[k];
        }
    }
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for schedule(static) if(runtime::openmp_kernel_loops_enabled())
#endif
    for (std::size_t i = 0; i < n; ++i) {
        epsilon(i, i) += Complex{1.0, 0.0};
    }

    MatrixComplex inv_eps = backend.inverse(epsilon);
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for schedule(static) if(runtime::openmp_kernel_loops_enabled())
#endif
    for (std::size_t i = 0; i < n; ++i) {
        inv_eps(i, i) -= Complex{1.0, 0.0};
    }

    // This is W = inv_eps^T * inv_v.
    return backend.gemm(inv_eps, inv_v, linalg::MatrixTranspose::Transpose, linalg::MatrixTranspose::NoTranspose);
}

Tensor3Real calculate_pq_ph_matrix(const MolecularIntegrals& integrals, const ParticleHoleBasis& ph_basis) {
    const std::size_t nmo = integrals.nmo();
    Tensor3Real out(nmo, nmo, ph_basis.size(), 0.0);
#if defined(GW_ENABLE_OPENMP_KERNEL_LOOPS)
#pragma omp parallel for collapse(3) schedule(static) if(runtime::openmp_kernel_loops_enabled())
#endif
    for (std::size_t ph = 0; ph < ph_basis.size(); ++ph) {
        for (std::size_t q = 0; q < nmo; ++q) {
            for (std::size_t p = 0; p < nmo; ++p) {
                const ParticleHolePair& ia = ph_basis[ph];
                out(p, q, ph) = integrals.eri(p, q, ia.i_occ, ia.a_mo);
            }
        }
    }
    return out;
}

} // namespace gw
