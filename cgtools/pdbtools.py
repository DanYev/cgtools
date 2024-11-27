import os
import sys
# import openmm as mm
from pathlib import Path
from pdbfixer.pdbfixer import PDBFixer
from openmm.app import PDBFile


AA_CODE_CONVERTER = {
    'A': 'ALA',
    'C': 'CYS',
    'D': 'ASP',
    'E': 'GLU',
    'F': 'PHE',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'K': 'LYS',
    'L': 'LEU',
    'M': 'MET',
    'N': 'ASN',
    'P': 'PRO',
    'Q': 'GLN',
    'R': 'ARG',
    'S': 'SER',
    'T': 'THR',
    'V': 'VAL',
    'W': 'TRP',
    'Y': 'TYR'
}


def convert_mutation_format(mutation):
    # Check if the input follows the expected format
    if len(mutation) < 3 or not mutation[0].isalpha() or not mutation[-1].isalpha() or not mutation[1:-1].isdigit():
        return "Invalid input format"
    
    # Extract components
    from_aa = mutation[0]
    to_aa = mutation[-1]
    position = mutation[1:-1]
    
    # Convert using the dictionary
    from_aa_3letter = AA_CODE_CONVERTER.get(from_aa, "UNK")
    to_aa_3letter = AA_CODE_CONVERTER.get(to_aa, "UNK")
    
    # Return the new format
    return f"{from_aa_3letter}-{position}-{to_aa_3letter}"


def prepare_aa_pdb(in_pdb, out_pdb, add_missing_atoms=False, add_hydrogens=False, variant=None):
    if variant:
        mutations = [convert_mutation_format(mutation) for mutation in variant]
    print(f"Opening {in_pdb}")
    pdb = PDBFixer(filename=in_pdb)
    if variant:
        print("Mutating residues")
        pdb.applyMutations(mutations, "A")
    print("Removing heterogens")
    pdb.removeHeterogens(False)
    print("Looking for missing residues")
    pdb.findMissingResidues()
    print("Looking for non-standard residues")
    pdb.findNonstandardResidues()
    print("Replacing non-standard residues")
    pdb.replaceNonstandardResidues()
    if add_missing_atoms:
        print("Looking for missing atoms")
        pdb.findMissingAtoms()
        print("Adding missing atoms")
        pdb.addMissingAtoms()
    if add_hydrogens:
        print("Adding missing hydrogens")
        pdb.addMissingHydrogens(7.0)
    topology = pdb.topology
    positions = pdb.positions
    print("Writing PDB")
    PDBFile.writeFile(topology, positions, open(out_pdb, 'w'))
    

def rename_chain(in_pdb, out_pdb, old_chain_id, new_chain_id):
    with open(in_pdb, 'r') as file:
        lines = file.readlines()
    updated_lines = []
    for line in lines:
        # PDB ATOM/HETATM lines have the chain ID in column 22
        if line.startswith(('ATOM', 'HETATM', )):
            if line[21] == old_chain_id:  # Column 22 (index 21) for chain ID
                line = line[:21] + new_chain_id + line[22:]
        updated_lines.append(line)
    with open(out_pdb, 'w') as file:
        file.writelines(updated_lines)
        
        
 import string

def rename_pdb_chains(input_pdb, output_pdb):
    """
    Rename chains in a PDB file in the order: uppercase letters, lowercase letters, digits.

    :param input_pdb: Path to the input PDB file.
    :param output_pdb: Path to the output PDB file.
    """
    # Define the order for renaming chains
    chain_order = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
    chain_mapping = {}  # Map original chain IDs to new chain IDs
    current_chain_index = 0

    with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
        for line in infile:
            # Only modify lines that define atomic coordinates
            if line.startswith(("ATOM", "HETATM", "TER")):
                original_chain_id = line[21]  # Chain ID is in column 22 (index 21)

                # Assign a new chain ID if this one hasn't been seen before
                if original_chain_id not in chain_mapping:
                    if current_chain_index >= len(chain_order):
                        raise ValueError("Too many chains in the PDB file to rename!")
                    chain_mapping[original_chain_id] = chain_order[current_chain_index]
                    current_chain_index += 1

                # Replace the chain ID in the line
                new_chain_id = chain_mapping[original_chain_id]
                line = line[:21] + new_chain_id + line[22:]

            # Write the (possibly modified) line to the output file
            outfile.write(line)

    print(f"Chains renamed and saved to {output_pdb}")


def extract_chain_names(pdb_file):
    """
    Extract a list of unique chain names from a PDB file.

    :param pdb_file: Path to the input PDB file.
    :return: List of unique chain names.
    """
    chain_names = set()

    with open(pdb_file, 'r') as file:
        for line in file:
            # Look for lines that define atomic coordinates
            if line.startswith(("ATOM", "HETATM", "TER")):
                chain_id = line[21].strip()  # Chain ID is in column 22 (index 21)
                if chain_id:  # Only add non-empty chain IDs
                    chain_names.add(chain_id)

    return sorted(chain_names)  # Return sorted list of unique chain names


    

    
    
