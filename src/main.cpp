#include "gw/gw.hpp"
#include "gw/io.hpp"
#include "gw/npy.hpp"
#include "gw/types.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

struct Cli {
    std::string input_dir{"."};
    std::string output_dir{"./gw_output"};
    std::size_t freq_points{200};
    std::size_t pade_params{16};
    std::optional<std::size_t> selected_state_1based{5};
    double eta{0.0};
};

void print_usage(const char* exe) {
    std::cerr << "Usage: " << exe << " [options]\n"
              << "Options:\n"
              << "  --input-dir PATH        Directory containing eri_mo.npy, mo_energy.npy, vxc_mo.npy, nocc.txt, fermi_energy.txt\n"
              << "  --freq-points N         Number of imaginary-frequency points [default: 200]\n"
              << "  --pade-params N         Number of Pade parameters [default: 16]\n"
              << "  --state N               1-based orbital index to calculate [default: 5]\n"
              << "  --all-states            Calculate all diagonal states\n"
              << "  --eta VALUE             Infinitesimal broadening [default: 0.0]\n"
              << "  --output-dir PATH       Directory containing E_c_before_Pade.txt and E_c.out \n"
              << "  --help                  Show this message\n";
}

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

} // namespace

int main(int argc, char** argv) {
    try {
        const Cli cli = parse_cli(argc, argv);

        std::cout << "\n--- Begin of C++20 G0W0 Calculation ---\n";
        std::cout << "Loading data from: " << cli.input_dir << '\n';

        gw::GwInput input;
        input.eri_mo = gw::read_tensor4_npy_f64(join_path(cli.input_dir, "eri_mo.npy"));
        input.mo_energy = gw::read_vector_npy_f64(join_path(cli.input_dir, "mo_energy.npy"));
        input.vxc_mo = gw::read_matrix_npy_f64(join_path(cli.input_dir, "vxc_mo.npy"));
        input.nocc = static_cast<std::size_t>(gw::read_int_text(join_path(cli.input_dir, "nocc.txt")));
        input.fermi_energy = gw::read_double_text(join_path(cli.input_dir, "fermi_energy.txt"));

        const std::size_t nmo = input.mo_energy.size();
        const std::size_t nvirt = nmo - input.nocc;
        const std::size_t n_ph = input.nocc * nvirt;

        std::cout << "  - Vxc MO basis shape: (" << input.vxc_mo.rows() << ", " << input.vxc_mo.cols() << ")\n";
        std::cout << "  - Fermi energy (eV): " << input.fermi_energy * gw::kHartreeToEv << '\n';
        std::cout << "  - Number of occupied orbitals: " << input.nocc << '\n';
        std::cout << "  - Number of virtual orbitals: " << nvirt << '\n';
        std::cout << "  - Number of MOs: " << nmo << '\n';
        std::cout << "  - Number of particle-hole states: " << n_ph << '\n';
        std::cout << "  - ERI MO shape: (" << input.eri_mo.dim0() << ", " << input.eri_mo.dim1() << ", "
                  << input.eri_mo.dim2() << ", " << input.eri_mo.dim3() << ")\n";

        gw::GwSettings settings;
        settings.num_freq_points_total = cli.freq_points;
        settings.num_pade_params = cli.pade_params;
        settings.eta = cli.eta;
        if (cli.selected_state_1based.has_value()) {
            if (*cli.selected_state_1based == 0) {
                throw std::runtime_error("--state must be 1-based and positive");
            }
            settings.selected_state_0based = *cli.selected_state_1based - 1;
        } else {
            settings.selected_state_0based.reset();
        }

        const gw::GwResult result = gw::run_g0w0(input, settings);

        const std::size_t output_state = settings.selected_state_0based.value_or(0);
        gw::output_self_energy_before_pade(cli.output_dir+"/E_c_before_Pade.txt", result.sigma_c_im_points, output_state);
        std::vector<gw::Complex> state_sigma(result.omegas.size());
        for (std::size_t f = 0; f < result.omegas.size(); ++f) {
            state_sigma[f] = result.sigma_c_im_points(output_state, f);
        }
        gw::output_self_energy_after_pade(cli.output_dir+"/E_c.out", state_sigma, result.omegas, settings.num_pade_params, 19836, input.fermi_energy);

        std::cout << "\n--- G0W0 Calculation Summary ---\n";
        if (settings.selected_state_0based.has_value()) {
            gw::output_final_energy_table(input.mo_energy, result.sigma_x, input.vxc_mo, result.qp_energy, {*settings.selected_state_0based});
        } else {
            std::vector<std::size_t> all_states(nmo);
            for (std::size_t i = 0; i < nmo; ++i) {
                all_states[i] = i;
            }
            gw::output_final_energy_table(input.mo_energy, result.sigma_x, input.vxc_mo, result.qp_energy, all_states);
        }
        std::cout << "\n--- End of C++20 G0W0 Calculation ---\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
