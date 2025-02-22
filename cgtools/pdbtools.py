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
## Classes and Functions ## 
###################################

class Atom:
    """
    Represents an ATOM or HETATM record from a PDB file.
    """
    def __init__(self, record, atid, name, alt_loc, resname, chid, resid,
                 icode, x, y, z, occupancy, bfactor, segid, element, charge):
        self.record = record        # "ATOM" or "HETATM"
        self.atid = atid            # Atom serial number
        self.name = name            # Atom name
        self.alt_loc = alt_loc      # Alternate location indicator
        self.resname = resname      # Residue name
        self.chid = chid            # Chain identifier
        self.resid = resid          # Residue sequence number
        self.icode = icode          # Insertion code
        self.x = x                  # x coordinate
        self.y = y                  # y coordinate
        self.z = z                  # z coordinate
        self.occupancy = occupancy  # Occupancy
        self.bfactor = bfactor      # Temperature factor
        self.segid = segid          # Segment identifier
        self.element = element      # Element symbol
        self.charge = charge        # Charge on the atom
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
        chid = line[21].strip()  # Updated to 'chid'
        resid = int(line[22:26])
        icode = line[26].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        occupancy_str = line[54:60].strip()
        occupancy = float(occupancy_str) if occupancy_str else None
        bfactor_str = line[60:66].strip()
        bfactor = float(bfactor_str) if bfactor_str else None
        # segid is typically found in columns 72-76
        segid = line[72:76].strip()
        element = line[76:78].strip()
        charge = line[78:80].strip()
        return cls(record, atid, name, alt_loc, resname, chid, resid,
                   icode, x, y, z, occupancy, bfactor, segid, element, charge)

    def __repr__(self):
        return (f"<Atom {self.record} {self.atid} {self.name} "
                f"{self.resname} {self.chid}{self.resid} seg:{self.segid} "
                f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f})>")

    def to_pdb_line(self):
        """
        Format the Atom instance as a fixed-width PDB ATOM/HETATM line.
        Adjust the formatting as needed.
        """
        return (
            f"{self.record:<6}"                  # record name, left-justified in 6 chars
            f"{self.atid:>5} "                   # atid number, right-justified in 5 chars + space
            f"{self.name:<4}"                    # atom name, left-justified in 4 chars
            f"{self.alt_loc:1}"                  # alternate location indicator in 1 char
            f"{self.resname:>3} "                # residue name, right-justified in 3 chars + space
            f"{self.chid:1}"                     # chain identifier in 1 char
            f"{self.resid:>4}"                   # residue sequence number, right-justified in 4 chars
            f"{self.icode:1}   "                 # insertion code in 1 char, then 3 spaces
            f"{self.x:>8.3f}"                    # x coordinate, 8 chars wide, 3 decimals
            f"{self.y:>8.3f}"                    # y coordinate, 8 chars wide, 3 decimals
            f"{self.z:>8.3f}"                    # z coordinate, 8 chars wide, 3 decimals
            f"{self.occupancy:>6.2f}"            # occupancy, 6 chars wide, 2 decimals
            f"{self.bfactor:>6.2f}"              # temp factor, 6 chars wide, 2 decimals
            f"{'':6}"                          # 6 spaces for columns 67-72
            f"{self.segid:>4}"                   # segid, right-justified in 4 chars (columns 73-76)
            f"{self.element:>2}"                 # element symbol, right-justified in 2 chars (columns 77-78)
            f"{self.charge:>2}"                  # charge, right-justified in 2 chars (columns 79-80)
        )


class AtomList(list):
    """
    AtomList is a helper subclass of the built-in list, specialized for storing Atom objects.
    
    Each element in an AtomList is expected to be an instance of the Atom class. AtomList
    provides convenient properties to access or update attributes across all Atom objects.
    
    Example:
        >>> # Assuming 'Atom' is defined and atoms are created from a PDB file:
        >>> atoms = AtomList([atom1, atom2, atom3])
        >>> print(atoms.names)  # Get a list of atom names
        ['CA', 'CB', 'CG']
        >>> atoms.xs = [1.0, 2.0, 3.0]  # Set new x coordinates for each atom
    """

    def __add__(self, other):
        """
        Implements addition of two AtomList objects.
        """
        return AtomList(super().__add__(other))

    @property
    def records(self):
        return [atom.record for atom in self]

    @records.setter
    def records(self, new_records):
        if len(new_records) != len(self):
            raise ValueError("Length of new records list must match the number of atoms")
        for i, rec in enumerate(new_records):
            self[i].record = rec

    @property
    def atids(self):
        return [atom.atid for atom in self]

    @atids.setter
    def atids(self, new_atids):
        if len(new_atids) != len(self):
            raise ValueError("Length of new atids list must match the number of atoms")
        for i, aid in enumerate(new_atids):
            self[i].atid = aid

    @property
    def names(self):
        return [atom.name for atom in self]

    @names.setter
    def names(self, new_names):
        if len(new_names) != len(self):
            raise ValueError("Length of new names list must match the number of atoms")
        for i, name in enumerate(new_names):
            self[i].name = name

    @property
    def alt_locs(self):
        return [atom.alt_loc for atom in self]

    @alt_locs.setter
    def alt_locs(self, new_alt_locs):
        if len(new_alt_locs) != len(self):
            raise ValueError("Length of new alt_locs list must match the number of atoms")
        for i, alt in enumerate(new_alt_locs):
            self[i].alt_loc = alt

    @property
    def resnames(self):
        return [atom.resname for atom in self]

    @resnames.setter
    def resnames(self, new_resnames):
        if len(new_resnames) != len(self):
            raise ValueError("Length of new resnames list must match the number of atoms")
        for i, rn in enumerate(new_resnames):
            self[i].resname = rn

    @property
    def chids(self):
        return [atom.chid for atom in self]

    @chids.setter
    def chids(self, new_chids):
        if len(new_chids) != len(self):
            raise ValueError("Length of new chids list must match the number of atoms")
        for i, cid in enumerate(new_chids):
            self[i].chid = cid

    @property
    def resids(self):
        return [atom.resid for atom in self]

    @resids.setter
    def resids(self, new_resids):
        if len(new_resids) != len(self):
            raise ValueError("Length of new resids list must match the number of atoms")
        for i, rid in enumerate(new_resids):
            self[i].resid = rid

    @property
    def icodes(self):
        return [atom.icode for atom in self]

    @icodes.setter
    def icodes(self, new_icodes):
        if len(new_icodes) != len(self):
            raise ValueError("Length of new icode list must match the number of atoms")
        for i, code in enumerate(new_icodes):
            self[i].icode = code

    @property
    def xs(self):
        return [atom.x for atom in self]

    @xs.setter
    def xs(self, new_xs):
        if len(new_xs) != len(self):
            raise ValueError("Length of new x coordinates must match the number of atoms")
        for i, x_val in enumerate(new_xs):
            self[i].x = x_val
            self[i].vec = (self[i].x, self[i].y, self[i].z)

    @property
    def ys(self):
        return [atom.y for atom in self]

    @ys.setter
    def ys(self, new_ys):
        if len(new_ys) != len(self):
            raise ValueError("Length of new y coordinates must match the number of atoms")
        for i, y_val in enumerate(new_ys):
            self[i].y = y_val
            self[i].vec = (self[i].x, self[i].y, self[i].z)

    @property
    def zs(self):
        return [atom.z for atom in self]

    @zs.setter
    def zs(self, new_zs):
        if len(new_zs) != len(self):
            raise ValueError("Length of new z coordinates must match the number of atoms")
        for i, z_val in enumerate(new_zs):
            self[i].z = z_val
            self[i].vec = (self[i].x, self[i].y, self[i].z)

    @property
    def occupancies(self):
        return [atom.occupancy for atom in self]

    @occupancies.setter
    def occupancies(self, new_occ):
        if len(new_occ) != len(self):
            raise ValueError("Length of new occupancy list must match the number of atoms")
        for i, occ in enumerate(new_occ):
            self[i].occupancy = occ

    @property
    def bfactors(self):
        return [atom.bfactor for atom in self]

    @bfactors.setter
    def bfactors(self, new_bfactors):
        if len(new_bfactors) != len(self):
            raise ValueError("Length of new bfactors list must match the number of atoms")
        for i, bf in enumerate(new_bfactors):
            self[i].bfactor = bf

    @property
    def segids(self):
        return [atom.segid for atom in self]

    @segids.setter
    def segids(self, new_segids):
        if len(new_segids) != len(self):
            raise ValueError("Length of new segids list must match the number of atoms")
        for i, seg in enumerate(new_segids):
            self[i].segid = seg

    @property
    def elements(self):
        return [atom.element for atom in self]

    @elements.setter
    def elements(self, new_elements):
        if len(new_elements) != len(self):
            raise ValueError("Length of new elements list must match the number of atoms")
        for i, elem in enumerate(new_elements):
            self[i].element = elem

    @property
    def charges(self):
        return [atom.charge for atom in self]

    @charges.setter
    def charges(self, new_charges):
        if len(new_charges) != len(self):
            raise ValueError("Length of new charges list must match the number of atoms")
        for i, ch in enumerate(new_charges):
            self[i].charge = ch

    @property
    def vecs(self):
        return [atom.vec for atom in self]

    @vecs.setter
    def vecs(self, new_vecs):
        if len(new_vecs) != len(self):
            raise ValueError("Length of new vectors list must match the number of atoms")
        for i, nvec in enumerate(new_vecs):
            self[i].vec = nvec

    def to_pdb_lines(self):
        """
        Convert all Atom objects in the list into their corresponding PDB formatted lines.
        """
        return [atom.to_pdb_line() for atom in self]

    def sort(self, key=None, reverse=False):
        """
        Sort the AtomList in place.
        
        Parameters:
            key (callable, optional): A function of one argument that is used to extract
                a comparison key from each list element. Defaults to sorting by
                (chid, resid, icode, atid).
            reverse (bool, optional): If True, the list elements are sorted as if each
                comparison were reversed.
        """
        # If no key is provided, use a default key function
        if key is None:
            key = lambda atom: (atom.chid, atom.resid, atom.icode, atom.atid)
        super().sort(key=key, reverse=reverse)

    def filter(self, filter_vals, mode="name"):
        """
        Filter atoms based on a given attribute.

        Parameters:
            filter_vals (iterable): An iterable of values to keep.
            mode (str): Which attribute to filter by. Valid modes include:
                        "record", "atid", "name", "alt_loc", "resname", "chid",
                        "resid", "icode", "x", "y", "z", "occupancy", "bfactor",
                        "segid", "element", "charge".
                        Default is "name".

        Returns:
            AtomList: A new AtomList containing only the atoms for which the chosen attribute is in filter_vals.
        """
        key_funcs = {
            "record": lambda atom: atom.record,
            "atid": lambda atom: atom.atid,
            "name": lambda atom: atom.name,
            "alt_loc": lambda atom: atom.alt_loc,
            "resname": lambda atom: atom.resname,
            "chid": lambda atom: atom.chid,
            "resid": lambda atom: atom.resid,
            "icode": lambda atom: atom.icode,
            "x": lambda atom: atom.x,
            "y": lambda atom: atom.y,
            "z": lambda atom: atom.z,
            "occupancy": lambda atom: atom.occupancy,
            "bfactor": lambda atom: atom.bfactor,
            "segid": lambda atom: atom.segid,
            "element": lambda atom: atom.element,
            "charge": lambda atom: atom.charge,
        }
        if mode not in key_funcs:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes are: {', '.join(key_funcs.keys())}")
        filter_vals = set(filter_vals)
        return AtomList([atom for atom in self if key_funcs[mode](atom) in filter_vals])

    def remove_atoms(self, atoms_to_remove):
        """
        Remove the specified atoms from the current list.
        """
        removal_set = set(atoms_to_remove)
        for atom in list(self):  # Create a copy to iterate over while removing
            if atom in removal_set:
                self.remove(atom)

    def save_pdb(self, filename):
        """
        Save the current AtomList instance to a PDB file.
        """
        with open(filename, "w") as f:
            for atom in self:
                f.write(atom.to_pdb_line() + "\n")
            f.write("END\n")



class Residue():
    """
    Represents a residue that holds a list of Atom objects.
    """
    def __init__(self, resname, resid, icode):
        self.resname = resname
        self.resid = resid
        self.icode = icode
        self._atoms = AtomList()  # List of Atom objects

    def add_atom(self, atom):
        self._atoms.append(atom)

    @property
    def atoms(self):
        """Return a list of all atoms in this residue."""
        return self._atoms

    def __iter__(self):
        return iter(self._atoms)

    def __repr__(self):
        return f"<Residue {self.resname} {self.resid}{self.icode} with {len(self._atoms)} atom(s)>"


class Chain():
    """
    Represents a chain that holds residues.
    """
    def __init__(self, chid):
        self.chid = chid
        # Residues keyed by (resid, icode)
        self.residues = {}

    def add_atom(self, atom):
        key = (atom.resid, atom.icode)
        if key not in self.residues:
            self.residues[key] = Residue(atom.resname, atom.resid, atom.icode)
        self.residues[key].add_atom(atom)

    @property
    def atoms(self):
        """Return a list of all atoms in this chain."""
        all_atoms = []
        # Sort residues by resid and insertion code for ordered iteration.
        for residue in sorted(self.residues.values(), key=lambda r: (r.resid, r.icode)):
            all_atoms.extend(residue.atoms)
        return AtomList(all_atoms)

    def __iter__(self):
        for residue in sorted(self.residues.values(), key=lambda r: (r.resid, r.icode)):
            yield residue

    def __repr__(self):
        return f"<Chain {self.chid} with {len(self.residues)} residue(s)>"


class Model():
    """
    Represents a model that holds chains.
    """
    def __init__(self, model_id):
        self.model_id = model_id
        # Chains keyed by chain identifier.
        self.chains = {}

    def __iter__(self):
        return iter(self.chains.values())

    def __repr__(self):
        return f"<Model {self.model_id} with {len(self.chains)} chain(s)>"

    def add_atom(self, atom):
        chid = atom.chid if atom.chid else ' '  # Use a blank chain id if not provided.
        if chid not in self.chains:
            self.chains[chid] = Chain(chid)
        self.chains[chid].add_atom(atom)

    @property
    def atoms(self):
        """Return a list of all atoms in this model."""
        all_atoms = []
        for chain in self.chains.values():
            all_atoms.extend(chain.atoms)
        return AtomList(all_atoms)   

    def select_chains(self, chids):
        """
        Return a list of chains based on given chids
        """
        return [chain for chid, chain in self.chains.items() if chid in chids]


class System():
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

    @property
    def atoms(self):
        """Return a list of all atoms in the system (from all models)."""
        all_atoms = []
        for model in self.models.values():
            all_atoms.extend(model.atoms)
        return AtomList(all_atoms)       

    def residues(self):
        """
        Generator that yields each residue from the system.
        Iterates through all models, chains, and residues in the system.
        """
        for model in self.models.values():
            # If needed, sort chains by chid for consistency.
            for chain in sorted(model.chains.values(), key=lambda c: c.chid):
                # Sort residues by resid and insertion code.
                for residue in sorted(chain.residues.values(), key=lambda r: (r.resid, r.icode)):
                    yield residue

    def chains(self):
        """
        Generator that yields each chain from the system.
        Iterates through all models and yields each chain contained within them..
        """
        for model in self.models.values():
            for chain in sorted(model.chains.values(), key=lambda c: c.chid):
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
                # Iterate over chains in sorted order by chid
                for chain in sorted(model.chains.values(), key=lambda c: c.chid):
                    # Iterate over residues in sorted order
                    for residue in sorted(chain.residues.values(), key=lambda r: (r.resid, r.icode)):
                        for atom in residue.atoms:
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

def parse_pdb(pdb_path):
    parser = PDBParser(pdb_path)
    system = parser.parse()
    return system


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
    

def rename_chain(in_pdb, out_pdb, old_chid, new_chid):
    with open(in_pdb, 'r') as file:
        lines = file.readlines()
    updated_lines = []
    for line in lines:
        # PDB ATOM/HETATM lines have the chain ID in column 22
        if line.startswith(('ATOM', 'HETATM', )):
            if line[21] == old_chid:  # Column 22 (index 21) for chain ID
                line = line[:21] + new_chid + line[22:]
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
                original_chid = line[21]  # Chain ID is in column 22 (index 21)

                # Assign a new chain ID if this one hasn't been seen before
                if original_chid not in chain_mapping:
                    if current_chain_index >= len(chain_order):
                        raise ValueError("Too many chains in the PDB file to rename!")
                    chain_mapping[original_chid] = chain_order[current_chain_index]
                    current_chain_index += 1

                # Replace the chain ID in the line
                new_chid = chain_mapping[original_chid]
                line = line[:21] + new_chid + line[22:]

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
                chid = line[21].strip()  # Chain ID is in column 22 (index 21)
                if chid:  # Only add non-empty chain IDs
                    chain_names.add(chid)

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
            chid = line[21]  # Extract chain ID (column 22, index 21)
            if chid not in chain_dict:
                chain_dict[chid] = []
            chain_dict[chid].append(line)
        else:
            # Keep non-ATOM, HETATM, and TER lines (e.g., HEADER, REMARK, END)
            other_lines.append(line)      
    # Sort chains alphabetically
    sorted_chids = sort_uld(chain_dict.keys())
    # Renumber atom IDs and write to the output file
    atom_id = 1  # Start atom ID renumbering
    with open(output_pdb, 'w') as file:
        # Write other (non-ATOM) lines first
        for line in other_lines:
            file.write(line)  
        # Write the sorted chains with updated atom IDs
        for chid in sorted_chids:
            for line in chain_dict[chid]:
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

    

    
    
