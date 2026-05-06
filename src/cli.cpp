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
              << "  --input-dir PATH        Directory containing eri_mo.npy, mo_energy.npy, vxc_mo.npy, nocc.txt, fermi_energy.txt\n"
              << "  --freq-points N         Number of imaginary-frequency points [default: 200]\n"
              << "  --pade-params N         Number of Pade parameters [default: 16]\n"
              << "  --state N               1-based orbital index to calculate [default: 5]\n"
              << "  --all-states            Calculate all diagonal states\n"
              << "  --eta VALUE             Infinitesimal broadening [default: 0.0]\n"
              << "  --linalg-backend NAME   Linear algebra backend: reference or blas-lapack [default: reference]\n"
              << "  --output-dir PATH       Directory containing E_c_before_Pade.out, E_c.out, and gw.out \n"
              << "  --help                  Show this message\n";
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
        } else if (arg == "--output-dir") {
            cli.output_dir = require_value(arg);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    std::filesystem::create_directories(cli.output_dir);
    return cli;
}

std::string join_path(const std::string& dir, const std::string& file) {
    return (std::filesystem::path(dir) / file).string();
}

} // namespace gw
