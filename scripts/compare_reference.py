#!/usr/bin/env python3

import argparse
import math
import re
import sys
from pathlib import Path


COMPLEX_TUPLE_RE = re.compile(
    r"""
    ^\s*
    \(?
    \s*
    (?P<real>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
    \s*
    [, \t]+
    \s*
    (?P<imag>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
    \s*
    \)?
    \s*$
    """,
    re.VERBOSE,
)


def read_complex_vector(path: Path) -> list[complex]:
    values: list[complex] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            match = COMPLEX_TUPLE_RE.match(stripped)
            if match is None:
                raise ValueError(
                    f"Could not parse complex value in {path} at line {line_number}: "
                    f"{stripped!r}"
                )

            real = float(match.group("real"))
            imag = float(match.group("imag"))
            values.append(complex(real, imag))

    return values


def compare_complex_vectors(
    reference: list[complex],
    output: list[complex],
    rtol: float,
    atol: float,
    compare_mode: str,
) -> bool:
    if len(reference) != len(output):
        print(f"FAILED: vector lengths differ: reference={len(reference)}, output={len(output)}")
        return False

    ok = True

    max_abs_complex = 0.0
    max_abs_real = 0.0
    max_abs_imag = 0.0
    max_rel_complex = 0.0

    worst_index = -1
    worst_ref = 0.0 + 0.0j
    worst_out = 0.0 + 0.0j
    worst_abs = 0.0
    worst_rel = 0.0

    for i, (ref, out) in enumerate(zip(reference, output)):
        abs_complex = abs(out - ref)
        abs_real = abs(out.real - ref.real)
        abs_imag = abs(out.imag - ref.imag)

        denom = max(abs(ref), 1.0)
        rel_complex = abs_complex / denom

        max_abs_complex = max(max_abs_complex, abs_complex)
        max_abs_real = max(max_abs_real, abs_real)
        max_abs_imag = max(max_abs_imag, abs_imag)
        max_rel_complex = max(max_rel_complex, rel_complex)

        if compare_mode == "complex":
            close = abs_complex <= (atol + rtol * abs(ref))
            abs_for_worst = abs_complex
            rel_for_worst = rel_complex
        elif compare_mode == "components":
            real_close = math.isclose(out.real, ref.real, rel_tol=rtol, abs_tol=atol)
            imag_close = math.isclose(out.imag, ref.imag, rel_tol=rtol, abs_tol=atol)
            close = real_close and imag_close
            abs_for_worst = max(abs_real, abs_imag)
            rel_for_worst = abs_for_worst / max(abs(ref.real), abs(ref.imag), 1.0)
        else:
            raise ValueError(f"Unknown compare mode: {compare_mode}")

        if abs_for_worst > worst_abs:
            worst_index = i
            worst_ref = ref
            worst_out = out
            worst_abs = abs_for_worst
            worst_rel = rel_for_worst

        if not close:
            ok = False

    if ok:
        print("OK")
        print(f"  entries          : {len(reference)}")
        print(f"  max_abs_complex  : {max_abs_complex:.6e}")
        print(f"  max_abs_real     : {max_abs_real:.6e}")
        print(f"  max_abs_imag     : {max_abs_imag:.6e}")
        print(f"  max_rel_complex  : {max_rel_complex:.6e}")
        return True

    print("FAILED")
    print(f"  entries          : {len(reference)}")
    print(f"  max_abs_complex  : {max_abs_complex:.6e}")
    print(f"  max_abs_real     : {max_abs_real:.6e}")
    print(f"  max_abs_imag     : {max_abs_imag:.6e}")
    print(f"  max_rel_complex  : {max_rel_complex:.6e}")
    print("  worst entry:")
    print(f"    index          : {worst_index}")
    print(f"    reference      : ({worst_ref.real:.17e},{worst_ref.imag:.17e})")
    print(f"    output         : ({worst_out.real:.17e},{worst_out.imag:.17e})")
    print(f"    abs_error      : {worst_abs:.6e}")
    print(f"    rel_error      : {worst_rel:.6e}")
    return False


ENERGY_COLUMNS = ("KS Energy", "Sigma_x", "Sigma_c", "Vxc", "QP Energy")


def read_energy_summary(log_file: Path, orbital: int) -> dict[str, float]:
    in_summary = False
    target = str(orbital)

    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "--- G0W0 Calculation Summary ---":
                in_summary = True
                continue

            if not in_summary:
                continue

            if stripped.startswith("--- End"):
                break

            parts = stripped.split()
            if len(parts) != 6 or parts[0] != target:
                continue

            return {name: float(value) for name, value in zip(ENERGY_COLUMNS, parts[1:])}

    raise ValueError(f"Could not find orbital {orbital} in {log_file}")


def compare_energy_summaries(
    reference: dict[str, float],
    output: dict[str, float],
    rtol: float,
    atol: float,
) -> bool:
    ok = True
    for name in ENERGY_COLUMNS:
        ref = reference[name]
        out = output[name]
        if not math.isclose(out, ref, rel_tol=rtol, abs_tol=atol):
            ok = False
            print(
                f"FAILED: {name}: output={out:.8g}, reference={ref:.8g}, "
                f"abs_error={abs(out - ref):.6e}"
            )

    if ok:
        print("OK")
        for name in ENERGY_COLUMNS:
            print(f"  {name:<10}: {output[name]:.8g}")
        return True

    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare GW reference output files."
    )

    parser.add_argument(
        "--comparison",
        choices=["complex-vector", "energy-summary"],
        default="complex-vector",
        help="Comparison type.",
    )

    parser.add_argument(
        "--reference-file",
        type=Path,
        required=True,
        help="Reference file.",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output file.",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-8,
        help="Relative tolerance.",
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-10,
        help="Absolute tolerance.",
    )

    parser.add_argument(
        "--compare-mode",
        choices=["complex", "components"],
        default="components",
        help=(
            "complex: compare |output-reference| as a complex norm; "
            "components: compare real and imaginary parts separately."
        ),
    )

    parser.add_argument(
        "--orbital",
        type=int,
        help="1-based orbital index for --comparison energy-summary.",
    )

    args = parser.parse_args()

    if not args.reference_file.exists():
        print(f"FAILED: reference file does not exist: {args.reference_file}")
        return 1

    if not args.output_file.exists():
        print(f"FAILED: output file does not exist: {args.output_file}")
        return 1

    try:
        if args.comparison == "complex-vector":
            reference = read_complex_vector(args.reference_file)
            output = read_complex_vector(args.output_file)

            ok = compare_complex_vectors(
                reference=reference,
                output=output,
                rtol=args.rtol,
                atol=args.atol,
                compare_mode=args.compare_mode,
            )
        elif args.comparison == "energy-summary":
            if args.orbital is None:
                raise ValueError("--orbital is required for --comparison energy-summary")

            reference = read_energy_summary(args.reference_file, args.orbital)
            output = read_energy_summary(args.output_file, args.orbital)

            ok = compare_energy_summaries(
                reference=reference,
                output=output,
                rtol=args.rtol,
                atol=args.atol,
            )
        else:
            raise ValueError(f"Unknown comparison type: {args.comparison}")

        return 0 if ok else 1

    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
