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
        def chain_sort_uld(x):
            return (x.isdigit(), x.islower(), x.isupper(), x)
        # If no key is provided, use a default key function
        if key is None:
            key = lambda atom: (chain_sort_uld(atom.chid), atom.resid, atom.icode, atom.atid)
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
        if isinstance(filter_vals, str):
            filter_vals = {filter_vals}
        if mode not in key_funcs:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes are: {', '.join(key_funcs.keys())}")
        filter_vals = set(filter_vals)
        return AtomList([atom for atom in self if key_funcs[mode](atom) in filter_vals])

    def renumber(self):
        """
        Renumber atoms in the list starting from 1
        """
        new_atids = [atid % 99999 for atid in range(1, len(self)+1)]
        self.atids = new_atids

    def remove_atoms(self, atoms_to_remove):
        """
        Remove the specified atoms from the current list.
        """
        removal_set = set(atoms_to_remove)
        for atom in list(self):  # Create a copy to iterate over while removing
            if atom in removal_set:
                self.remove(atom)

    def read_pdb(self, in_pdb):
        """
        Read a PDB file to an AtomList instance.
        """
        with open(in_pdb, 'r') as file:
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
                        self.append(atom)
                    except Exception as e:
                        print(f"Error parsing line: {line.strip()} -> {e}")
                elif record_type == "ENDMDL":
                    current_model = 1

    def write_pdb(self, out_pdb, append=False):
        """
        Save the current AtomList instance to a PDB file.
        """
        mode = 'a' if append else 'w'
        with open(out_pdb, mode) as f:
            for atom in self:
                f.write(atom.to_pdb_line() + "\n")

    def write_ndx(self, filename, header='[ group ]', append=False, wrap=15):
        """
        Write the atom IDs (atids) of the AtomList to a Gromacs .ndx file.

        Parameters:
            filename (str): The path to the output .ndx file.
            header (str): The header for the group. Default is '[ group ]'.
            append (bool): If True, append to the file; otherwise, overwrite it.
            wrap (int): Number of atids per line in the output file.
        """
        mode = 'a' if append else 'w'
        atids = [str(atid) for atid in self.atids]
        with open(filename, mode) as f:
            f.write(f"{header}\n") # Write header
            for i in range(0, len(self), wrap):  # Write atids wrapping every `wrap` elements
                line = " ".join(atids[i:i+wrap])
                f.write(line + "\n") 
            f.write("\n")  # Add an extra newline for separation between groups if appending


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
        # for residue in sorted(self.residues.values(), key=lambda r: (r.resid, r.icode)):
        for residue in self.residues.values():
            all_atoms.extend(residue.atoms)
        return AtomList(all_atoms)

    def __iter__(self):
        for residue in self.residues.values():
            yield residue

    def __repr__(self):
        return f"<Chain {self.chid} with {len(self.residues)} residue(s)>"


class Model():
    """
    Represents a model that holds chains.
    """
    def __init__(self, modid):
        self.modid = modid
        # Chains keyed by chain identifier.
        self.chains = {}

    def __iter__(self):
        return iter(self.chains.values())

    def __repr__(self):
        return f"<Model {self.modid} with {len(self.chains)} chain(s)>"

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

    def add_atom(self, atom, modid=1):
        if modid not in self.models:
            self.models[modid] = Model(modid)
        self.models[modid].add_atom(atom)

    def add_atoms(self, atoms, modid=1):
        if modid not in self.models:
            self.models[modid] = Model(modid)
        for atom in atoms:
            self.models[modid].add_atom(atom)

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

    def write_pdb(self, filename):
        """
        Save the current System instance to a PDB file.
        Writes MODEL/ENDMDL records if multiple models exist.
        """
        with open(filename, "w") as f:
            # Sort models by modid
            sorted_modids = sorted(self.models.keys())
            multiple_models = len(sorted_modids) > 1
            for modid in sorted_modids:
                model = self.models[modid]
                if multiple_models:
                    f.write(f"MODEL     {modid}\n")
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
                        system.add_atom(atom, modid=current_model)
                    except Exception as e:
                        print(f"Error parsing line: {line.strip()} -> {e}")
                elif record_type == "ENDMDL":
                    current_model = 1
        return system

###################################
## Higher Level Functions ## 
###################################

def pdb2system(pdb_path) -> System:
    """
    Read a pdb into a System object
    """
    parser = PDBParser(pdb_path)
    system = parser.parse()
    return system


def pdb2atomlist(pdb_path) -> AtomList:
    """
    Read a pdb into an AtomList object
    """
    atoms = AtomList()
    atoms.read_pdb(pdb_path)
    return atoms    


def sort_chains_atoms(atoms):
    """
    Sort an AtomList and renumber atom IDs from 1 to 99999
    """
    atoms.sort()
    new_atids = [atid % 99999 for atid in range(1, len(atoms)+1)]
    atoms.atids = new_atids


def rename_chains_for_gromacs(atoms):
    """
    Rename chains in a AtomList in the order: uppercase letters, lowercase letters, digits.
    """
    import string
    # Define the order for renaming chains
    new_chids = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
    curr_chid = atoms[0].chid
    counter = 0
    for atom in atoms:
        if atom.chid != curr_chid:
            curr_chid = atom.chid
            counter += 1
        atom.chid = new_chids[counter]      


def sort_pdb(in_pdb, out_pdb):
    """
    Sort PDB file to make GROMACS happy (hopefully)
    """
    system = parse_pdb(in_pdb)
    atoms = system.atoms
    sort_chains_atoms(atoms)
    rename_chains_for_gromacs(atoms)
    atoms.write_pdb(out_pdb)
    print(f"Chains and atoms sorted, renamed and saved to {out_pdb}") 


def clean_pdb(in_pdb, out_pdb, add_missing_atoms=False, add_hydrogens=False, pH=7.0):
    print(f"Processing {in_pdb}", file=sys.stderr)
    pdb = PDBFixer(filename=in_pdb)
    print("Removing heterogens, Looking for missing residues", file=sys.stderr)
    pdb.removeHeterogens(False)
    pdb.findMissingResidues()
    print("Replacing non-standard residues", file=sys.stderr)
    pdb.findNonstandardResidues()
    pdb.replaceNonstandardResidues()
    if add_missing_atoms:
        print("Adding missing atoms", file=sys.stderr)
        pdb.findMissingAtoms()
        pdb.addMissingAtoms()
    if add_hydrogens:
        print("Adding missing hydrogens", file=sys.stderr)
        pdb.addMissingHydrogens(pH)
    topology = pdb.topology
    positions = pdb.positions
    PDBFile.writeFile(topology, positions, open(out_pdb, 'w'))
    print(f"Written PDB to {out_pdb}", file=sys.stderr)


def rename_chain_in_pdb(in_pdb, new_chain_id):
    system = parse_pdb(in_pdb)
    atoms = system.atoms   
    new_chids = [new_chain_id for atom in atoms]
    atoms.chids = new_chids
    atoms.write_pdb(in_pdb)


def write_ndx(atoms, fpath='system.ndx', backbone_atoms=("CA", "P", "C1'")):
    in_pdb = 'test.pdb'
    system = parse_pdb(in_pdb)
    atoms = system.atoms
    atoms.write_ndx(fpath, header=f'[ System ]', append=False, wrap=15) # sys ndx
    backbone = atoms.filter(backbone_atoms, mode='name')
    backbone.write_ndx(fpath, header=f'[ Backbone ]', append=True, wrap=15) # bb ndx
    chids = sorted(set(atoms.chids))
    for chid in chids:
        selected_atoms = atoms.filter(chid, mode='chid')
        selected_atoms.write_ndx(fpath, header=f'[ chain_{chid} ]', append=True, wrap=15) # chain ndx
        print(f"Written PDB to {fpath}", file=sys.stderr)


def update_bfactors(in_pdb, out_pdb, bfactors):
    """
    Update bfactors in a PDB file
    :param input_pdb: Path to the input PDB file.
    :param output_pdb: Path to the output PDB file.
    :param bfactors: list or ndarray
    """
    b_factors = read_b_factors(b_factor_file)
    update_pdb_b_factors(pdb_file, b_factors, output_file)
    print(f"Updated PDB file saved as '{output_file}'.")    

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


    

    
    
