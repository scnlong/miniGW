#include "gw/cli.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace gw {
namespace {

void print_usage(const char* exe) {
    std::cerr << "Usage: " << exe << " [options]\n"
              << "Options:\n"
              << "  --input-dir PATH             Directory containing eri_mo.npy, mo_energy.npy, vxc_mo.npy, nocc.txt, fermi_energy.txt\n"
              << "  --freq-points N              Number of imaginary-frequency points [default: 200]\n"
              << "  --pade-params N              Number of Pade parameters [default: 16]\n"
              << "  --state N                    1-based orbital index to calculate [default: 5]\n"
              << "  --all-states                 Calculate all diagonal states\n"
              << "  --eta VALUE                  Infinitesimal broadening [default: 0.0]\n"
              << "  --linalg-backend NAME        reference, blas-lapack, scalapack, cosma, or cublas [default: blas-lapack]\n"
              << "  --frequency-parallel MODE    auto, serial, mpi, or openmp [default: auto]\n"
              << "  --kernel-parallel MODE       auto, serial, or openmp for local kernel loops [default: auto]\n"
              << "  --print-memory-footprint     Print an algorithmic memory estimate before running GW\n"
              << "  --output-dir PATH            Directory containing E_c_before_Pade.out, E_c.out, and gw.out\n"
              << "  --help                       Show this message\n";
}

} // namespace

Cli parse_cli(int argc, char** argv) {
    Cli cli;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };
        if (arg == "--input-dir") {
            cli.input_dir = require_value(arg);
        } else if (arg == "--freq-points") {
            cli.freq_points = static_cast<std::size_t>(std::stoull(require_value(arg)));
        } else if (arg == "--pade-params") {
            cli.pade_params = static_cast<std::size_t>(std::stoull(require_value(arg)));
        } else if (arg == "--state") {
            cli.selected_state_1based = static_cast<std::size_t>(std::stoull(require_value(arg)));
        } else if (arg == "--all-states") {
            cli.selected_state_1based.reset();
        } else if (arg == "--eta") {
            cli.eta = std::stod(require_value(arg));
        } else if (arg == "--linalg-backend") {
            cli.linalg_backend = require_value(arg);
        } else if (arg == "--frequency-parallel") {
            cli.frequency_parallel = require_value(arg);
        } else if (arg == "--kernel-parallel") {
            cli.kernel_parallel = require_value(arg);
        } else if (arg == "--print-memory-footprint") {
            cli.print_memory_footprint = true;
        } else if (arg == "--output-dir") {
            cli.output_dir = require_value(arg);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }
    std::filesystem::create_directories(cli.output_dir);

    if (cli.freq_points == 0) {
        throw std::runtime_error("--freq-points must be positive");
    }
    if (cli.pade_params == 0) {
        throw std::runtime_error("--pade-params must be positive");
    }
    if (cli.frequency_parallel != "auto" && cli.frequency_parallel != "serial" &&
        cli.frequency_parallel != "mpi" && cli.frequency_parallel != "openmp") {
        throw std::runtime_error("--frequency-parallel must be one of: auto, serial, mpi, openmp");
    }
    if (cli.kernel_parallel != "auto" && cli.kernel_parallel != "serial" &&
        cli.kernel_parallel != "openmp") {
        throw std::runtime_error("--kernel-parallel must be one of: auto, serial, openmp");
    }
    return cli;
}

std::string join_path(const std::string& dir, const std::string& file) {
    return (std::filesystem::path(dir) / file).string();
}

} // namespace gw
