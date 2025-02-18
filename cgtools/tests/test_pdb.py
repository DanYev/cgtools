class Atom:
    """
    Represents an ATOM or HETATM record from a PDB file.
    """
    def __init__(self, record, serial, name, alt_loc, res_name, chain_id, res_seq,
                 i_code, x, y, z, occupancy, temp_factor, element, charge):
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
        self.temp_factor = temp_factor  # Temperature factor
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
        temp_factor_str = line[60:66].strip()
        temp_factor = float(temp_factor_str) if temp_factor_str else None
        element = line[76:78].strip()
        charge = line[78:80].strip()
        return cls(record, serial, name, alt_loc, res_name, chain_id, res_seq,
                   i_code, x, y, z, occupancy, temp_factor, element, charge)

    def __repr__(self):
        return (f"<Atom {self.record} {self.serial} {self.name} "
                f"{self.res_name} {self.chain_id}{self.res_seq} "
                f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f})>")


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


# Example usage:
if __name__ == "__main__":
    pdb_path = "chain_ds.pdb"  # Replace with your PDB file path
    parser = PDBParser(pdb_path)
    system = parser.parse()

    print(system)  # System summary

    # Iterate over models:
    for model in system:
        print(model)
        # Iterate over chains in each model:
        for chain in model:
            print(" ", chain)
            # Iterate over residues in each chain:
            for residue in chain:
                print("    ", residue)
                # List atoms in each residue:
                for atom in residue.atoms():
                    print("       ", atom)

    # Alternatively, get a flat list of all atoms in the system:
    all_atoms = system.atoms()
    print(f"\nTotal atoms parsed: {len(all_atoms)}")
