from cgtools.pdbtools import PDBParser, Atom, AtomList, System, Model, Chain, Residue


def test_read_pdb():
    pdb_path = "dsRNA.pdb"  
    parser = PDBParser(pdb_path)
    system = parser.parse()
    print(system)  # System summary
    atoms = system.atoms
    print(f"\nTotal atoms parsed: {len(atoms)}")


def test_save_pdb():
    pdb_path = "dsRNA.pdb" 
    parser = PDBParser(pdb_path)
    system = parser.parse()
    system.save_pdb("test.pdb")


def test_mask(mask=["P", "C3'"]):
    pdb_path = "dsRNA.pdb" 
    parser = PDBParser(pdb_path)
    system = parser.parse()
    atoms = system.atoms.mask(mask)
    new_bs = range(len(atoms))
    atoms.b_factors = new_bs
    atoms.save_pdb("test.pdb")
    

if __name__ == "__main__":
    # test_read_pdb()     
    # test_save_pdb() 
    test_mask()              