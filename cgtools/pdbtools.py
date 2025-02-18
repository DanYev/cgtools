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


class Atom:
    """
    Represents an ATOM or HETATM record from a PDB file.
    """
    def __init__(self, record, serial, name, alt_loc, res_name, chain_id, res_seq,
                 i_code, x, y, z, occupancy, b_factor, element, charge):
        self.record = record          # "ATOM" or "HETATM"
        self.serial = serial          # Atom serial number
        self.name = name              # Atom name
        self.alt_loc = alt_loc        # Alternate location indicator
        self.res_name = res_name      # Residue name
        self.chain_id = chain_id      # Chain identifier
        self.res_seq = res_seq        # Residue sequence number
        self.i_code = i_code          # Insertion code
        self.x = x                    # x coordinate
        self.y = y                    # y coordinate
        self.z = z                    # z coordinate
        self.occupancy = occupancy    # Occupancy
        self.b_factor = b_factor      # Temperature factor
        self.element = element        # Element symbol
        self.charge = charge          # Charge on the atom

    @classmethod
    def from_pdb_line(cls, line):
        """
        Parse a line from a PDB file that starts with 'ATOM' or 'HETATM'
        and return an Atom instance.
        """
        record = line[0:6].strip()
        serial = int(line[6:11])
        name = line[12:16].strip()
        alt_loc = line[16].strip()
        res_name = line[17:20].strip()
        chain_id = line[21].strip()
        res_seq = int(line[22:26])
        i_code = line[26].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        occupancy_str = line[54:60].strip()
        occupancy = float(occupancy_str) if occupancy_str else None
        b_factor_str = line[60:66].strip()
        b_factor = float(b_factor_str) if b_factor_str else None
        element = line[76:78].strip()
        charge = line[78:80].strip()
        return cls(record, serial, name, alt_loc, res_name, chain_id, res_seq,
                   i_code, x, y, z, occupancy, b_factor, element, charge)

    def __repr__(self):
        return (f"<Atom {self.record} {self.serial} {self.name} "
                f"{self.res_name} {self.chain_id}{self.res_seq} "
                f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f})>")


    def to_pdb_line(self):
        """
        Format the Atom instance as a fixed-width PDB ATOM/HETATM line.
        Adjust the formatting as needed.
        """
        return (
            f"{self.record:<6}"  # record name left-justified in 6 chars
            f"{self.serial:>5} "  # serial number right-justified in 5 chars + space
            f"{self.name:<4}"     # atom name left-justified in 4 chars
            f"{self.alt_loc:1}"   # alternate location indicator in 1 char
            f"{self.res_name:>3} " # residue name right-justified in 3 chars + space
            f"{self.chain_id:1}"  # chain identifier in 1 char
            f"{self.res_seq:>4}"   # residue sequence number right-justified in 4 chars
            f"{self.i_code:1}   " # insertion code in 1 char, then 3 spaces
            f"{self.x:>8.3f}"     # x coordinate, 8 chars wide, 3 decimals
            f"{self.y:>8.3f}"     # y coordinate, 8 chars wide, 3 decimals
            f"{self.z:>8.3f}"     # z coordinate, 8 chars wide, 3 decimals
            f"{self.occupancy:>6.2f}"  # occupancy, 6 chars wide, 2 decimals
            f"{self.b_factor:>6.2f}          "  # temp factor, 6 chars wide, 2 decimals, plus 10 spaces for alignment
            f"{self.element:>2}"  # element symbol right-justified in 2 chars
            f"{self.charge:>2}"   # charge right-justified in 2 chars
        )


class Residue:
    """
    Represents a residue that holds a list of Atom objects.
    """
    def __init__(self, res_name, res_seq, i_code):
        self.res_name = res_name
        self.res_seq = res_seq
        self.i_code = i_code
        self._atoms = []  # List of Atom objects

    def add_atom(self, atom):
        self._atoms.append(atom)

    def atoms(self):
        """Return a list of all atoms in this residue."""
        return self._atoms

    def __iter__(self):
        return iter(self._atoms)

    def __repr__(self):
        return f"<Residue {self.res_name} {self.res_seq}{self.i_code} with {len(self._atoms)} atom(s)>"


class Chain:
    """
    Represents a chain that holds residues.
    """
    def __init__(self, chain_id):
        self.chain_id = chain_id
        # Residues keyed by (res_seq, i_code)
        self.residues = {}

    def add_atom(self, atom):
        key = (atom.res_seq, atom.i_code)
        if key not in self.residues:
            self.residues[key] = Residue(atom.res_name, atom.res_seq, atom.i_code)
        self.residues[key].add_atom(atom)

    def atoms(self):
        """Return a list of all atoms in this chain."""
        all_atoms = []
        # Sort residues by res_seq and insertion code for ordered iteration.
        for residue in sorted(self.residues.values(), key=lambda r: (r.res_seq, r.i_code)):
            all_atoms.extend(residue.atoms())
        return all_atoms

    def __iter__(self):
        for residue in sorted(self.residues.values(), key=lambda r: (r.res_seq, r.i_code)):
            yield residue

    def __repr__(self):
        return f"<Chain {self.chain_id} with {len(self.residues)} residue(s)>"


class Model:
    """
    Represents a model that holds chains.
    """
    def __init__(self, model_id):
        self.model_id = model_id
        # Chains keyed by chain identifier.
        self.chains = {}

    def add_atom(self, atom):
        chain_id = atom.chain_id if atom.chain_id else ' '  # Use a blank chain id if not provided.
        if chain_id not in self.chains:
            self.chains[chain_id] = Chain(chain_id)
        self.chains[chain_id].add_atom(atom)

    def atoms(self):
        """Return a list of all atoms in this model."""
        all_atoms = []
        for chain in self.chains.values():
            all_atoms.extend(chain.atoms())
        return all_atoms

    def __iter__(self):
        return iter(self.chains.values())

    def __repr__(self):
        return f"<Model {self.model_id} with {len(self.chains)} chain(s)>"


class System:
    """
    Represents the entire system that holds models.
    """
    def __init__(self):
        # Models keyed by model id (default id = 1 if no MODEL record is provided)
        self.models = {}

    def add_atom(self, atom, model_id=1):
        if model_id not in self.models:
            self.models[model_id] = Model(model_id)
        self.models[model_id].add_atom(atom)

    def atoms(self):
        """Return a list of all atoms in the system (from all models)."""
        all_atoms = []
        for model in self.models.values():
            all_atoms.extend(model.atoms())
        return all_atoms

    def __iter__(self):
        return iter(self.models.values())

    def __repr__(self):
        return f"<System with {len(self.models)} model(s)>"

    def save_pdb(self, filename):
        """
        Save the current System instance to a PDB file.
        Writes MODEL/ENDMDL records if multiple models exist.
        """
        with open(filename, "w") as f:
            # Sort models by model_id
            sorted_model_ids = sorted(self.models.keys())
            multiple_models = len(sorted_model_ids) > 1

            for model_id in sorted_model_ids:
                model = self.models[model_id]
                if multiple_models:
                    f.write(f"MODEL     {model_id}\n")
                # Iterate over chains in sorted order by chain_id
                for chain in sorted(model.chains.values(), key=lambda c: c.chain_id):
                    # Iterate over residues in sorted order
                    for residue in sorted(chain.residues.values(), key=lambda r: (r.res_seq, r.i_code)):
                        for atom in residue.atoms():
                            f.write(atom.to_pdb_line() + "\n")
                if multiple_models:
                    f.write("ENDMDL\n")
            f.write("END\n")


class PDBParser:
    """
    Parses a PDB file and builds the hierarchical structure using composition.
    """
    def __init__(self, pdb_file):
        self.pdb_file = pdb_file

    def parse(self):
        """
        Parse the PDB file and return a System instance.
        """
        system = System()
        current_model = 1  # Default model id

        with open(self.pdb_file, 'r') as file:
            for line in file:
                record_type = line[0:6].strip()
                if record_type == "MODEL":
                    try:
                        current_model = int(line[10:14].strip())
                    except ValueError:
                        current_model = 1
                elif record_type in ("ATOM", "HETATM"):
                    try:
                        atom = Atom.from_pdb_line(line)
                        system.add_atom(atom, model_id=current_model)
                    except Exception as e:
                        print(f"Error parsing line: {line.strip()} -> {e}")
                elif record_type == "ENDMDL":
                    current_model = 1
        return system


#########################
#### FUNCTIONS ##########
#########################

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
    
    
def read_b_factors(file_path):
    """
    Reads the B-factor file and returns a dictionary mapping residue numbers to B-factors.
    """
    b_factors = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) != 2:
                continue
            residue_number = int(parts[0])
            b_factor = float(parts[1])
            b_factors[residue_number] = b_factor
    return b_factors


def update_bfactors(pdb_file, b_factors, output_file):
    """
    Reads the PDB file, updates B-factors for each residue, and writes the updated PDB file.
    """
    with open(pdb_file, 'r') as file:
        lines = file.readlines()
    lines = [line for line in lines if line.startswith("ATOM")]
    updated_lines = []
    idx = 0
    previous_residue_number = int(lines[0][22:26].strip())
    for line in lines:
        residue_number = int(line[22:26].strip())
        if residue_number != previous_residue_number:
            idx += 1
        b_factor = b_factors[idx]
        updated_line = f"{line[:60]}{b_factor:6.2f}{line[66:]}"
        updated_lines.append(updated_line)
        previous_residue_number = residue_number
    with open(output_file, 'w') as file:
        file.writelines(updated_lines)


def set_bfactors():
    pdb_file = "input.pdb"       # Replace with the path to your PDB file
    b_factor_file = "b_factors.txt"  # Replace with the path to your B-factor file
    output_file = "output.pdb"   # Path to save the updated PDB file
    
    if not os.path.exists(pdb_file):
        print(f"PDB file '{pdb_file}' does not exist.")
        return
    if not os.path.exists(b_factor_file):
        print(f"B-factor file '{b_factor_file}' does not exist.")
        return
    
    b_factors = read_b_factors(b_factor_file)
    update_pdb_b_factors(pdb_file, b_factors, output_file)
    print(f"Updated PDB file saved as '{output_file}'.")    




    

    
    
