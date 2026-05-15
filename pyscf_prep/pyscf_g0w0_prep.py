import numpy as np
from typing import List
from pyscf import gto, dft, ao2mo
from collections import defaultdict
from math import ceil
import sys

# Define conversion factor from Hartree to eV
HARTREE_TO_EV = 27.211386245981

# --- 1. Define the molecule ---
# Example: Water molecule
mol = gto.M(
    atom = 'O 0 0 0; H 0.7571 0.0000 0.5861; H -0.7571 0.0000 0.5861',
    basis = 'def2-svp',  # A minimal basis set for demonstration
    charge = 0,
    spin = 0,
    verbose = 0, # Suppress PySCF verbose output
)

# --- 2. Save the basis to FHI-aims format ---
def pyscf_to_fhi_aims_gaussian_format(atom_symbol: str, pyscf_basis: List) -> str:
    lines = []
    for shell in pyscf_basis:
        ang_mom = shell[0]
        primitives = shell[1:]
        if len(primitives) == 1:
            # Uncontracted
            lines.append(f"gaussian {ang_mom} 1 {primitives[0][0]:.12g}")
        else:
            # Contracted shell
            lines.append(f"gaussian {ang_mom} {len(primitives)}")
            for i in primitives:
                lines.append(f"  {i[0]:.10f}   {i[1]:.10f}")
    return "\n".join(lines)

with open("pyscf_basis_to_aims.txt", "w") as f:
    all_atom_symbols = []
    for i in range(mol.natm):
        all_atom_symbols.append(mol.atom_symbol(i))
    all_atom_symbols = list(set(all_atom_symbols))
    f.write("Don't forget to add ---> include_min_basis false <--- in the species, and comment out NAO.\n")
    for i in all_atom_symbols:
        basis_data = mol._basis[i]
        f.write(f"Element: {i} \n")
        f.write("pure_gauss .true.\n")
        f.write(pyscf_to_fhi_aims_gaussian_format(i, basis_data)+"\n")
        f.write("\n")

# --- 3. Perform a Restricted Kohn-Sham (RKS) DFT calculation ---
# Using PBE functional
mf = dft.RKS(mol)
mf.xc = 'pbe' # Exchange-correlation functional
mf.kernel()

# Check if calculation converged
if mf.converged:
    print("DFT calculation converged successfully.")
else:
    raise RuntimeError("DFT calculation did not converge")

# --- 4. Extract eigenvalues (orbital energies) and eigenvectors (MO coefficients) ---
mo_energy = mf.mo_energy # Kohn-Sham orbital energies
mo_coeff = mf.mo_coeff   # Molecular orbital coefficients (eigenvectors)

print(f"\nNumber of molecular orbitals: {mo_energy.shape[0]}")
print(f"Molecular orbital energies (Ha): {mo_energy}")
# Convert and print in eV
print(f"Molecular orbital energies (eV): {mo_energy * HARTREE_TO_EV}")
print(f"Shape of MO coefficients: {mo_coeff.shape}")

# --- 5. Save the MO to Turbomole mos file format ---
# AO and MO data
ao_labels = mol.ao_labels()
nmo = mo_coeff.shape[1]
nsaos_lines=ceil(nmo / 4)

# Write to Turbomole-style mos file
with open("mos", "w") as f:
    f.write("$scfmo   expanded   format(4d20.14)\n")
    for iorb in range(nmo):
        eigval = mo_energy[iorb]
        if eigval >= 0:
            str_tmp = f"{eigval:.14E}"
        else:
            str_tmp = f"{eigval:.13E}"
        f.write(f"{iorb+1:6d}  a      eigenvalue={str_tmp}   nsaos={nmo}\n".replace('E', 'D'))
        coeffs = mo_coeff[:, iorb]
        for i in range(nsaos_lines):
            chunk = coeffs[i*4:(i+1)*4]
            line=''
            for x in chunk:
                if x >=0 :
                    line += f"{x:.14E}".replace('E', 'D')
                else:
                    line += f"{x:.13E}".replace('E', 'D')
            f.write(line + "\n")
    f.write("$end\n")

# --- 6. Calculate Fermi energy ---
# For finite systems, Fermi energy is typically between HOMO and LUMO.
# nocc is the number of occupied orbitals
nocc = mol.nelectron // 2 # For restricted (RKS), number of electrons / 2 gives occupied orbitals
nvirt = mo_energy.shape[0] - nocc # Number of virtual orbitals

if nocc > 0 and nocc < mo_energy.shape[0]:
    homo_energy = mo_energy[nocc - 1]
    lumo_energy = mo_energy[nocc]
    fermi_energy = (homo_energy + lumo_energy) / 2.0
    print(f"\nNumber of occupied orbitals: {nocc}")
    print(f"HOMO energy (Ha): {homo_energy:.6f}")
    print(f"LUMO energy (Ha): {lumo_energy:.6f}")
    print(f"Fermi energy (Ha): {fermi_energy:.6f}")
    # Convert and print in eV
    print(f"HOMO energy (eV): {homo_energy * HARTREE_TO_EV:.6f}")
    print(f"LUMO energy (eV): {lumo_energy * HARTREE_TO_EV:.6f}")
    print(f"Fermi energy (eV): {fermi_energy * HARTREE_TO_EV:.6f}")
else:
    fermi_energy = 0.0 # Or handle as appropriate for very small/large systems
    print("\nCould not determine clear HOMO/LUMO for Fermi energy calculation.")


# --- 7. Transform two-electron integrals (ERI) from AO to MO basis ---
# The ERI in PySCF is typically in (ij|kl) notation, where i,j,k,l are basis functions.
# We need (pq|rs) in physicist's notation for G0W0, where p,q,r,s are MOs.
# PySCF's ao2mo.kernel returns (pq|rs) in chemist's notation (pr|qs) by default.
# We need to reshape and transpose to get physicist's notation (pq|rs).
# (p,q,r,s) correspond to MO indices.

print("\nTransforming two-electron integrals (ERI) to MO basis...")
# Get ERI in AO basis
eri_ao = mol.intor('int2e')

print("\nTransforming overlap matrix of MO basis...")
S_ao = mol.intor('int1e_ovlp')
mo_overlap = mo_coeff.T @ S_ao @ mo_coeff

# Transform ERI from AO to MO basis using mo_coeff
# The result eri_mo_chemist is (pr|qs) in chemist's notation
eri_mo_chemist = ao2mo.kernel(eri_ao, mo_coeff, compact=False)

# Reshape to 4D tensor (nmo, nmo, nmo, nmo)
eri_mo_chemist = eri_mo_chemist.reshape(nmo, nmo, nmo, nmo)

# Convert from chemist's notation (pr|qs) to physicist's notation (pq|rs)
# (pr|qs) means integral over (p(1)r(2) | q(1)s(2))
# (pq|rs) means integral over (p(1)q(1) | r(2)s(2))
# This involves transposing the second and third indices: (p, r, q, s) -> (p, q, r, s)
# So, eri_mo[p,q,r,s] = eri_mo_chemist[p,r,q,s]
#eri_mo = np.transpose(eri_mo_chemist, (0, 2, 1, 3))
eri_mo = eri_mo_chemist

print(f"Shape of ERI in MO basis (physicist's notation): {eri_mo.shape}")


# --- 8. Calculate and transform Vxc (Exchange-Correlation Potential) to MO basis ---
print("\nCalculating and transforming Vxc to MO basis...")
# For DFT, Vxc = V_effective - V_Hartree
# mf.get_veff() returns the total effective potential (including V_Hartree and V_xc)
# mf.get_j() returns the Hartree (Coulomb) potential
vxc_ao = mf.get_veff() - mf.get_j()

# Transform Vxc from AO to MO basis: V_xc_MO = C.T @ V_xc_AO @ C
vxc_mo = np.dot(mo_coeff.T, np.dot(vxc_ao, mo_coeff))

print(f"Shape of Vxc in MO basis: {vxc_mo.shape}")


# --- 9. Save the extracted data to one HDF5 file ---
output_path = "gw_input.h5"

# Store all arrays in C-order float64 layout, matching miniGW's row-major Matrix/Tensor classes.
with h5py.File(output_path, "w") as h5:
    h5.attrs["format"] = "miniGW PySCF G0W0 input"
    h5.attrs["format_version"] = 1
    h5.attrs["energy_unit"] = "Hartree"
    h5.attrs["nocc"] = np.int64(nocc)
    h5.attrs["fermi_energy"] = np.float64(fermi_energy)
    h5.create_dataset("mo_energy", data=np.asarray(mo_energy, dtype=np.float64), compression=None)
    h5.create_dataset("eri_mo", data=np.ascontiguousarray(eri_mo, dtype=np.float64), compression=None)
    h5.create_dataset("vxc_mo", data=np.ascontiguousarray(vxc_mo, dtype=np.float64), compression=None)

print("\nData saved successfully:")
print(f"- {output_path}")
print("  datasets: /mo_energy, /eri_mo, /vxc_mo")
print("  attributes: nocc, fermi_energy, energy_unit, format_version")
