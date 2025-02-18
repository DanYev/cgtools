from cgtools.pdbtools import PDBParser


def test_read_pdb():
    pdb_path = "chain_ds.pdb"  
    parser = PDBParser(pdb_path)
    system = parser.parse()
    print(system)  # System summary
    atoms = system.atoms()
    print(f"\nTotal atoms parsed: {len(atoms)}")


def test_save_pdb():
    pdb_path = "chain_ds.pdb" 
    parser = PDBParser(pdb_path)
    system = parser.parse()
    for atom in system.atoms():
        atom.b_factor = 1
    system.save_pdb('saved.pdb')


# Example usage:
if __name__ == "__main__":
    test_read_pdb()     
    test_save_pdb()               