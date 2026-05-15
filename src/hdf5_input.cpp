#include "hdf5_input.hpp"

#include <hdf5.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gw {
namespace {

class H5Object {
public:
    H5Object() = default;
    H5Object(hid_t id, herr_t (*closer)(hid_t)) : id_(id), closer_(closer) {}

    H5Object(const H5Object&) = delete;
    H5Object& operator=(const H5Object&) = delete;

    H5Object(H5Object&& other) noexcept : id_(other.id_), closer_(other.closer_) {
        other.id_ = -1;
        other.closer_ = nullptr;
    }

    H5Object& operator=(H5Object&& other) noexcept {
        if (this != &other) {
            close();
            id_ = other.id_;
            closer_ = other.closer_;
            other.id_ = -1;
            other.closer_ = nullptr;
        }
        return *this;
    }

    ~H5Object() { close(); }

    [[nodiscard]] hid_t get() const noexcept { return id_; }

private:
    hid_t id_{-1};
    herr_t (*closer_)(hid_t){nullptr};

    void close() noexcept {
        if (id_ >= 0 && closer_ != nullptr) {
            closer_(id_);
        }
        id_ = -1;
        closer_ = nullptr;
    }
};

[[nodiscard]] std::string dims_to_string(const std::vector<hsize_t>& dims) {
    std::ostringstream out;
    out << '(';
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << dims[i];
    }
    out << ')';
    return out.str();
}

[[nodiscard]] std::size_t checked_dim(hsize_t value, const std::string& name) {
    if (value > static_cast<hsize_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("HDF5 dataset dimension too large for size_t: " + name);
    }
    return static_cast<std::size_t>(value);
}

[[nodiscard]] std::vector<hsize_t> dataset_shape(hid_t dataset, const std::string& name) {
    H5Object space(H5Dget_space(dataset), H5Sclose);
    if (space.get() < 0) {
        throw std::runtime_error("Could not get dataspace for HDF5 dataset: " + name);
    }
    const int rank = H5Sget_simple_extent_ndims(space.get());
    if (rank < 0) {
        throw std::runtime_error("Could not get rank for HDF5 dataset: " + name);
    }
    std::vector<hsize_t> dims(static_cast<std::size_t>(rank));
    if (rank > 0 && H5Sget_simple_extent_dims(space.get(), dims.data(), nullptr) < 0) {
        throw std::runtime_error("Could not get shape for HDF5 dataset: " + name);
    }
    return dims;
}

[[nodiscard]] std::vector<double> read_f64_dataset_raw(hid_t file, const std::string& name, int expected_rank) {
    H5Object dataset(H5Dopen2(file, name.c_str(), H5P_DEFAULT), H5Dclose);
    if (dataset.get() < 0) {
        throw std::runtime_error("Could not open HDF5 dataset: " + name);
    }
    const std::vector<hsize_t> dims = dataset_shape(dataset.get(), name);
    if (static_cast<int>(dims.size()) != expected_rank) {
        throw std::runtime_error("HDF5 dataset " + name + " has rank " + std::to_string(dims.size())
                                 + ", expected " + std::to_string(expected_rank));
    }

    std::size_t count = 1;
    for (const hsize_t dim : dims) {
        const std::size_t sdim = checked_dim(dim, name);
        if (sdim != 0 && count > std::numeric_limits<std::size_t>::max() / sdim) {
            throw std::runtime_error("HDF5 dataset is too large: " + name);
        }
        count *= sdim;
    }

    std::vector<double> values(count);
    if (count > 0 && H5Dread(dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) < 0) {
        throw std::runtime_error("Could not read HDF5 float64 dataset: " + name);
    }
    return values;
}

[[nodiscard]] std::vector<double> read_vector_f64(hid_t file, const std::string& name) {
    return read_f64_dataset_raw(file, name, 1);
}

[[nodiscard]] MatrixReal read_matrix_f64(hid_t file, const std::string& name) {
    H5Object dataset(H5Dopen2(file, name.c_str(), H5P_DEFAULT), H5Dclose);
    if (dataset.get() < 0) {
        throw std::runtime_error("Could not open HDF5 dataset: " + name);
    }
    const std::vector<hsize_t> dims = dataset_shape(dataset.get(), name);
    if (dims.size() != 2) {
        throw std::runtime_error("HDF5 dataset " + name + " has shape " + dims_to_string(dims) + ", expected a matrix");
    }
    const std::size_t rows = checked_dim(dims[0], name);
    const std::size_t cols = checked_dim(dims[1], name);
    std::vector<double> values(rows * cols);
    if (!values.empty() && H5Dread(dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) < 0) {
        throw std::runtime_error("Could not read HDF5 float64 matrix: " + name);
    }
    return MatrixReal(rows, cols, std::move(values));
}

[[nodiscard]] Tensor4Real read_tensor4_f64(hid_t file, const std::string& name) {
    H5Object dataset(H5Dopen2(file, name.c_str(), H5P_DEFAULT), H5Dclose);
    if (dataset.get() < 0) {
        throw std::runtime_error("Could not open HDF5 dataset: " + name);
    }
    const std::vector<hsize_t> dims = dataset_shape(dataset.get(), name);
    if (dims.size() != 4) {
        throw std::runtime_error("HDF5 dataset " + name + " has shape " + dims_to_string(dims) + ", expected a rank-4 tensor");
    }
    const std::size_t n0 = checked_dim(dims[0], name);
    const std::size_t n1 = checked_dim(dims[1], name);
    const std::size_t n2 = checked_dim(dims[2], name);
    const std::size_t n3 = checked_dim(dims[3], name);
    std::vector<double> values(n0 * n1 * n2 * n3);
    if (!values.empty() && H5Dread(dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) < 0) {
        throw std::runtime_error("Could not read HDF5 float64 rank-4 tensor: " + name);
    }
    return Tensor4Real(n0, n1, n2, n3, std::move(values));
}

template <typename T>
[[nodiscard]] T read_scalar_attribute(hid_t file, const std::string& name, hid_t native_type) {
    H5Object attr(H5Aopen(file, name.c_str(), H5P_DEFAULT), H5Aclose);
    if (attr.get() < 0) {
        throw std::runtime_error("Could not open HDF5 file attribute: " + name);
    }
    T value{};
    if (H5Aread(attr.get(), native_type, &value) < 0) {
        throw std::runtime_error("Could not read HDF5 file attribute: " + name);
    }
    return value;
}

void validate_input(const GwInput& input) {
    const std::size_t nmo = input.mo_energy.size();
    if (nmo == 0) {
        throw std::runtime_error("HDF5 input error: mo_energy is empty");
    }
    if (input.nocc == 0 || input.nocc >= nmo) {
        throw std::runtime_error("HDF5 input error: nocc must be in [1, nmo-1]");
    }
    if (input.vxc_mo.rows() != nmo || input.vxc_mo.cols() != nmo) {
        throw std::runtime_error("HDF5 input error: vxc_mo shape does not match mo_energy length");
    }
    if (input.eri_mo.dim0() != nmo || input.eri_mo.dim1() != nmo || input.eri_mo.dim2() != nmo || input.eri_mo.dim3() != nmo) {
        throw std::runtime_error("HDF5 input error: eri_mo shape does not match mo_energy length");
    }
}

} // namespace

GwInput read_gw_input_hdf5(const std::string& path) {
    H5Object file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    if (file.get() < 0) {
        throw std::runtime_error("Could not open HDF5 input file: " + path);
    }

    GwInput input;
    input.mo_energy = read_vector_f64(file.get(), "/mo_energy");
    input.eri_mo = read_tensor4_f64(file.get(), "/eri_mo");
    input.vxc_mo = read_matrix_f64(file.get(), "/vxc_mo");

    const long long nocc_i64 = read_scalar_attribute<long long>(file.get(), "nocc", H5T_NATIVE_LLONG);
    if (nocc_i64 < 0) {
        throw std::runtime_error("HDF5 input error: nocc attribute is negative");
    }
    input.nocc = static_cast<std::size_t>(nocc_i64);
    input.fermi_energy = read_scalar_attribute<double>(file.get(), "fermi_energy", H5T_NATIVE_DOUBLE);

    validate_input(input);
    return input;
}

} // namespace gw
