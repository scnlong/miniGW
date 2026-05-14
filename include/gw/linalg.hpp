#pragma once

#include "gw/types.hpp"

#include <memory>
#include <string_view>
#include <vector>

namespace gw::linalg {

enum class MatrixTranspose {
    NoTranspose,
    Transpose,
    ConjugateTranspose
};

enum class BackendFamily {
    LocalHost,
    DistributedHost,
    Device
};

struct BackendCapabilities {
    BackendFamily family{BackendFamily::LocalHost};
    bool thread_safe{true};
    bool uses_internal_threads{false};
    bool distributed_mpi{false};
    bool uses_device_memory{false};

    // True for the COSMA ScaLAPACK-compatible distributed GEMM provider.
    // This is not the same as miniGW's own device-resident cuBLAS/HIP backend:
    // matrices are still managed by the distributed ScaLAPACK-style host path,
    // but GEMM calls are routed explicitly to COSMA's prefixed PBLAS ABI
    // symbol, e.g. cosma_pzgemm_.
    bool uses_cosma_pxgemm{false};

    // True if the external provider may use GPUs internally.  This flag is
    // diagnostic only; it does not imply that miniGW owns device buffers or that
    // a given run actually exercised GPU kernels.
    bool external_provider_may_use_gpu{false};
};

class Backend {
public:
    virtual ~Backend() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual BackendCapabilities capabilities() const noexcept { return {}; }

    [[nodiscard]] virtual MatrixComplex inverse(MatrixComplex a) const = 0;

    [[nodiscard]] virtual MatrixComplex gemm(const MatrixComplex& a,
                                             const MatrixComplex& b,
                                             MatrixTranspose trans_a = MatrixTranspose::NoTranspose,
                                             MatrixTranspose trans_b = MatrixTranspose::NoTranspose) const = 0;

    [[nodiscard]] virtual std::vector<Complex> gemv(const MatrixComplex& a,
                                                    const std::vector<double>& x,
                                                    MatrixTranspose trans_a = MatrixTranspose::NoTranspose) const = 0;

    [[nodiscard]] virtual Complex quadratic_form(const std::vector<double>& x,
                                                 const MatrixComplex& a) const = 0;
};

class ReferenceBackend final : public Backend {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] MatrixComplex inverse(MatrixComplex a) const override;
    [[nodiscard]] MatrixComplex gemm(const MatrixComplex& a,
                                     const MatrixComplex& b,
                                     MatrixTranspose trans_a = MatrixTranspose::NoTranspose,
                                     MatrixTranspose trans_b = MatrixTranspose::NoTranspose) const override;
    [[nodiscard]] std::vector<Complex> gemv(const MatrixComplex& a,
                                            const std::vector<double>& x,
                                            MatrixTranspose trans_a = MatrixTranspose::NoTranspose) const override;
    [[nodiscard]] Complex quadratic_form(const std::vector<double>& x,
                                         const MatrixComplex& a) const override;
};

[[nodiscard]] const Backend& reference_backend();
[[nodiscard]] std::shared_ptr<const Backend> make_reference_backend();

} // namespace gw::linalg
