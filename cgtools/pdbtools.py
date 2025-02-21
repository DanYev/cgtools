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


###################################
## Generic Classes and Functions ## 
###################################

class Atom:
    """
    Represents an ATOM or HETATM record from a PDB file.
    """
    def __init__(self, record, atid, name, alt_loc, resname, chain_id, resid,
                 icode, x, y, z, occupancy, b_factor, element, charge):
        self.record = record          # "ATOM" or "HETATM"
        self.atid = atid          # Atom atid number
        self.name = name              # Atom name
        self.alt_loc = alt_loc        # Alternate location indicator
        self.resname = resname      # Residue name
        self.chain_id = chain_id      # Chain identifier
        self.resid = resid        # Residue sequence number
        self.icode = icode          # Insertion code
        self.x = x                    # x coordinate
        self.y = y                    # y coordinate
        self.z = z                    # z coordinate
        self.occupancy = occupancy    # Occupancy
        self.b_factor = b_factor      # Temperature factor
        self.element = element        # Element symbol
        self.charge = charge          # Charge on the atom
        self.vec = (self.x, self.y, self.z)

    @classmethod
    def from_pdb_line(cls, line):
        """
        Parse a line from a PDB file that starts with 'ATOM' or 'HETATM'
        and return an Atom instance.
        """
        record = line[0:6].strip()
        atid = int(line[6:11])
        name = line[12:16].strip()
        alt_loc = line[16].strip()
        resname = line[17:20].strip()
        chain_id = line[21].strip()
        resid = int(line[22:26])
        icode = line[26].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        occupancy_str = line[54:60].strip()
        occupancy = float(occupancy_str) if occupancy_str else None
        b_factor_str = line[60:66].strip()
        b_factor = float(b_factor_str) if b_factor_str else None
        element = line[76:78].strip()
        charge = line[78:80].strip()
        return cls(record, atid, name, alt_loc, resname, chain_id, resid,
                   icode, x, y, z, occupancy, b_factor, element, charge)

    def __repr__(self):
        return (f"<Atom {self.record} {self.atid} {self.name} "
                f"{self.resname} {self.chain_id}{self.resid} "
                f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f})>")

    def to_pdb_line(self):
        """
        Format the Atom instance as a fixed-width PDB ATOM/HETATM line.
        Adjust the formatting as needed.
        """
        return (
            f"{self.record:<6}"  # record name left-justified in 6 chars
            f"{self.atid:>5} "  # atid number right-justified in 5 chars + space
            f"{self.name:<4}"     # atom name left-justified in 4 chars
            f"{self.alt_loc:1}"   # alternate location indicator in 1 char
            f"{self.resname:>3} " # residue name right-justified in 3 chars + space
            f"{self.chain_id:1}"  # chain identifier in 1 char
            f"{self.resid:>4}"   # residue sequence number right-justified in 4 chars
            f"{self.icode:1}   " # insertion code in 1 char, then 3 spaces
            f"{self.x:>8.3f}"     # x coordinate, 8 chars wide, 3 decimals
            f"{self.y:>8.3f}"     # y coordinate, 8 chars wide, 3 decimals
            f"{self.z:>8.3f}"     # z coordinate, 8 chars wide, 3 decimals
            f"{self.occupancy:>6.2f}"  # occupancy, 6 chars wide, 2 decimals
            f"{self.b_factor:>6.2f}          "  # temp factor, 6 chars wide, 2 decimals, plus 10 spaces for alignment
            f"{self.element:>2}"  # element symbol right-justified in 2 chars
            f"{self.charge:>2}"   # charge right-justified in 2 chars
        )


class AtomList(list):
    """
    AtomList is a helper subclass of the built-in list, specialized for storing Atom objects.
    
    Each element in an AtomList is expected to be an instance of the Atom class. AtomList
    provides convenient properties to access or update attributes across all Atom objects
    in the list. For example, you can get or set the x coordinates for all atoms at once.
    
    Example:
        >>> # Assuming 'Atom' is already defined and atoms are created from a PDB file:
        >>> atoms = AtomList([atom1, atom2, atom3])
        >>> print(atoms.names)  # Get a list of atom names
        ['CA', 'CB', 'CG']
        >>> atoms.x = [1.0, 2.0, 3.0]  # Set new x coordinates for each atom
    """

    def __add__(self, other):
        """
        Implements addition of two AtomList objects.
        
        Parameters:
            other (AtomList): Another AtomList to concatenate.
            
        Returns:
            AtomList: A new AtomList containing atoms from both self and other.
        """
        return AtomList(super().__add__(other))

    @property
    def records(self):
        """Return a list of record strings (e.g., 'ATOM' or 'HETATM') for all atoms."""
        return [atom.record for atom in self]

    @records.setter
    def records(self, new_records):
        if len(new_records) != len(self):
            raise ValueError("Length of new records list must match the number of atoms")
        for i, rec in enumerate(new_records):
            self[i].record = rec

    @property
    def atids(self):
        """Return a list of atom id numbers for all atoms."""
        return [atom.atid for atom in self]

    @atids.setter
    def atids(self, new_atids):
        if len(new_atids) != len(self):
            raise ValueError("Length of new atids list must match the number of atoms")
        for i, aid in enumerate(new_atids):
            self[i].atid = aid

    @property
    def names(self):
        """Return a list of atom names for all atoms."""
        return [atom.name for atom in self]

    @names.setter
    def names(self, new_names):
        if len(new_names) != len(self):
            raise ValueError("Length of new names list must match the number of atoms")
        for i, name in enumerate(new_names):
            self[i].name = name

    @property
    def alt_locs(self):
        """Return a list of alternate location indicators for all atoms."""
        return [atom.alt_loc for atom in self]

    @alt_locs.setter
    def alt_locs(self, new_alt_locs):
        if len(new_alt_locs) != len(self):
            raise ValueError("Length of new alt_locs list must match the number of atoms")
        for i, alt in enumerate(new_alt_locs):
            self[i].alt_loc = alt

    @property
    def resnames(self):
        """Return a list of residue names for all atoms."""
        return [atom.resname for atom in self]

    @resnames.setter
    def resnames(self, new_resnames):
        if len(new_resnames) != len(self):
            raise ValueError("Length of new resnames list must match the number of atoms")
        for i, rn in enumerate(new_resnames):
            self[i].resname = rn

    @property
    def chain_ids(self):
        """Return a list of chain identifiers for all atoms."""
        return [atom.chain_id for atom in self]

    @chain_ids.setter
    def chain_ids(self, new_chain_ids):
        if len(new_chain_ids) != len(self):
            raise ValueError("Length of new chain_ids list must match the number of atoms")
        for i, cid in enumerate(new_chain_ids):
            self[i].chain_id = cid

    @property
    def resids(self):
        """Return a list of residue sequence numbers for all atoms."""
        return [atom.resid for atom in self]

    @resids.setter
    def resids(self, new_resids):
        if len(new_resids) != len(self):
            raise ValueError("Length of new resids list must match the number of atoms")
        for i, rid in enumerate(new_resids):
            self[i].resid = rid

    @property
    def icode(self):
        """Return a list of insertion codes for all atoms."""
        return [atom.icode for atom in self]

    @icode.setter
    def icode(self, new_icode):
        if len(new_icode) != len(self):
            raise ValueError("Length of new icode list must match the number of atoms")
        for i, code in enumerate(new_icode):
            self[i].icode = code

    @property
    def x(self):
        """Return a list of x coordinates for all atoms."""
        return [atom.x for atom in self]

    @x.setter
    def x(self, new_x):
        if len(new_x) != len(self):
            raise ValueError("Length of new x coordinates must match the number of atoms")
        for i, x_val in enumerate(new_x):
            self[i].x = x_val
            # Update the vector tuple as well.
            self[i].vec = (self[i].x, self[i].y, self[i].z)

    @property
    def y(self):
        """Return a list of y coordinates for all atoms."""
        return [atom.y for atom in self]

    @y.setter
    def y(self, new_y):
        if len(new_y) != len(self):
            raise ValueError("Length of new y coordinates must match the number of atoms")
        for i, y_val in enumerate(new_y):
            self[i].y = y_val
            self[i].vec = (self[i].x, self[i].y, self[i].z)

    @property
    def z(self):
        """Return a list of z coordinates for all atoms."""
        return [atom.z for atom in self]

    @z.setter
    def z(self, new_z):
        if len(new_z) != len(self):
            raise ValueError("Length of new z coordinates must match the number of atoms")
        for i, z_val in enumerate(new_z):
            self[i].z = z_val
            self[i].vec = (self[i].x, self[i].y, self[i].z)

    @property
    def occupancy(self):
        """Return a list of occupancy values for all atoms."""
        return [atom.occupancy for atom in self]

    @occupancy.setter
    def occupancy(self, new_occ):
        if len(new_occ) != len(self):
            raise ValueError("Length of new occupancy list must match the number of atoms")
        for i, occ in enumerate(new_occ):
            self[i].occupancy = occ

    @property
    def b_factors(self):
        """Return a list of temperature factors (B-factors) for all atoms."""
        return [atom.b_factor for atom in self]

    @b_factors.setter
    def b_factors(self, new_bf):
        if len(new_bf) != len(self):
            raise ValueError("Length of new b_factors list must match the number of atoms")
        for i, bf in enumerate(new_bf):
            self[i].b_factor = bf

    @property
    def elements(self):
        """Return a list of element symbols for all atoms."""
        return [atom.element for atom in self]

    @elements.setter
    def elements(self, new_elements):
        if len(new_elements) != len(self):
            raise ValueError("Length of new elements list must match the number of atoms")
        for i, elem in enumerate(new_elements):
            self[i].element = elem

    @property
    def charges(self):
        """Return a list of charge strings for all atoms."""
        return [atom.charge for atom in self]

    @charges.setter
    def charges(self, new_charges):
        if len(new_charges) != len(self):
            raise ValueError("Length of new charges list must match the number of atoms")
        for i, ch in enumerate(new_charges):
            self[i].charge = ch

    def to_pdb_lines(self):
        """
        Convert all Atom objects in the list into their corresponding PDB formatted lines.
        
        Returns:
            list: A list of strings, each representing an Atom in PDB format.
        """
        return [atom.to_pdb_line() for atom in self]

    def __repr__(self):
        return f"<AtomList with {len(self)} atoms>"


class Residue(AtomList):
    """
    Represents a residue that holds a list of Atom objects.
    """
    def __init__(self, resname, resid, icode):
        self.resname = resname
        self.resid = resid
        self.icode = icode
        self._atoms = []  # List of Atom objects

    def add_atom(self, atom):
        self._atoms.append(atom)

    def atoms(self):
        """Return a list of all atoms in this residue."""
        return self._atoms

    def __iter__(self):
        return iter(self._atoms)

    def __repr__(self):
        return f"<Residue {self.resname} {self.resid}{self.icode} with {len(self._atoms)} atom(s)>"


class Chain(AtomList):
    """
    Represents a chain that holds residues.
    """
    def __init__(self, chain_id):
        self.chain_id = chain_id
        # Residues keyed by (resid, icode)
        self.residues = {}

    def add_atom(self, atom):
        key = (atom.resid, atom.icode)
        if key not in self.residues:
            self.residues[key] = Residue(atom.resname, atom.resid, atom.icode)
        self.residues[key].add_atom(atom)

    def atoms(self):
        """Return a list of all atoms in this chain."""
        all_atoms = []
        # Sort residues by resid and insertion code for ordered iteration.
        for residue in sorted(self.residues.values(), key=lambda r: (r.resid, r.icode)):
            all_atoms.extend(residue.atoms())
        return all_atoms

    def __iter__(self):
        for residue in sorted(self.residues.values(), key=lambda r: (r.resid, r.icode)):
            yield residue

    def __repr__(self):
        return f"<Chain {self.chain_id} with {len(self.residues)} residue(s)>"


class Model(AtomList):
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


class System(AtomList):
    """
    Represents the entire system that holds models.
    """
    def __init__(self):
        # Models keyed by model id (default id = 1 if no MODEL record is provided)
        self.models = {}

    def __iter__(self):
        return iter(self.models.values())

    def __repr__(self):
        return f"<System with {len(self.models)} model(s)>"

    def add_atom(self, atom, model_id=1):
        if model_id not in self.models:
            self.models[model_id] = Model(model_id)
        self.models[model_id].add_atom(atom)

    def add_atoms(self, atoms, model_id=1):
        if model_id not in self.models:
            self.models[model_id] = Model(model_id)
        for atom in atoms:
            self.models[model_id].add_atom(atom)

    def atoms(self):
        """Return a list of all atoms in the system (from all models)."""
        all_atoms = []
        for model in self.models.values():
            all_atoms.extend(model.atoms())
        return all_atoms        

    def residues(self):
        """
        Generator that yields each residue from the system.
        Iterates through all models, chains, and residues in the system.
        """
        for model in self.models.values():
            # If needed, sort chains by chain_id for consistency.
            for chain in sorted(model.chains.values(), key=lambda c: c.chain_id):
                # Sort residues by resid and insertion code.
                for residue in sorted(chain.residues.values(), key=lambda r: (r.resid, r.icode)):
                    yield residue

    def chains(self):
        """
        Generator that yields each chain from the system.
        Iterates through all models and yields each chain contained within them..
        """
        for model in self.models.values():
            for chain in sorted(model.chains.values(), key=lambda c: c.chain_id):
                yield chain

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
                    for residue in sorted(chain.residues.values(), key=lambda r: (r.resid, r.icode)):
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


###################################
## Higher Level Functions ## 
###################################


def clean_pdb(in_pdb, out_pdb, add_missing_atoms=False, add_hydrogens=False, pH=7.0):
    print(f"Opening {in_pdb}")
    pdb = PDBFixer(filename=in_pdb)
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
        pdb.addMissingHydrogens(pH)
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


def rearrange_chains_and_renumber_atoms(input_pdb, output_pdb):
    """
    Rearrange chains in a PDB file alphabetically by chain ID and renumber atom IDs.

    Parameters:
        input_pdb (str): Path to the input PDB file.
        output_pdb (str): Path to save the rearranged PDB file.
    """
    with open(input_pdb, 'r') as file:
        pdb_lines = file.readlines()
    # Group lines by chain ID
    chain_dict = {}
    other_lines = []
    for line in pdb_lines:
        if line.startswith(("ATOM", )):
            chain_id = line[21]  # Extract chain ID (column 22, index 21)
            if chain_id not in chain_dict:
                chain_dict[chain_id] = []
            chain_dict[chain_id].append(line)
        else:
            # Keep non-ATOM, HETATM, and TER lines (e.g., HEADER, REMARK, END)
            other_lines.append(line)      
    # Sort chains alphabetically
    sorted_chain_ids = sort_uld(chain_dict.keys())
    # Renumber atom IDs and write to the output file
    atom_id = 1  # Start atom ID renumbering
    with open(output_pdb, 'w') as file:
        # Write other (non-ATOM) lines first
        for line in other_lines:
            file.write(line)  
        # Write the sorted chains with updated atom IDs
        for chain_id in sorted_chain_ids:
            for line in chain_dict[chain_id]:
                if line.startswith(("ATOM", )):
                    # Update the atom ID (columns 7-11, index 6-11)
                    updated_line = f"{line[:6]}{atom_id:5d}{line[11:]}"
                    file.write(updated_line)
                    atom_id += 1  # Increment atom ID
                    if atom_id > 99999:
                        atom_id = 1
                else:
                    # Write TER lines as-is
                    file.write(line)  


################################################################################
# Helper functions
################################################################################  

def sort_uld(alist):
    """
    Sorts characters in a list such that they appear in the following order: 
    uppercase letters first, then lowercase letters, followed by digits. 
    Helps with orgazing gromacs multichain files
    """
    slist = sorted(alist, key=lambda x: (x.isdigit(), x.islower(), x.isupper(), x))
    return slist


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

    

    
    
