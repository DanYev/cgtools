import os 
from cgtools.pdbtools import PDBParser, Atom, AtomList, System, Model, Chain, Residue, parse_pdb

TEST_PDB = "dsRNA.pdb" 

def test_read_pdb():
    pdb_path = TEST_PDB
    parser = PDBParser(pdb_path)
    system = parser.parse()
    atoms = system.atoms
    if_passed = len(atoms) > 0
    assert if_passed


def test_save_system():
    pdb_path = TEST_PDB 
    test_pdb = "test.pdb"
    if test_pdb in os.listdir():
        os.remove(test_pdb)
    system = parse_pdb(pdb_path)
    system.save_pdb(test_pdb)
    if_passed = test_pdb in os.listdir()
    assert if_passed


def test_save_atoms():
    pdb_path = TEST_PDB
    test_pdb = "test.pdb"
    if test_pdb in os.listdir():
        os.remove(test_pdb)
    system = parse_pdb(pdb_path)
    atoms = system.atoms
    atoms.save_pdb(test_pdb)
    if_passed = test_pdb in os.listdir()
    assert if_passed


def test_chain():
    pdb_path = TEST_PDB
    system = parse_pdb(pdb_path)
    model = system.models[1]
    chain_a = model.chains["A"]
    chain_list = model.select_chains(["A", "B"])
    if_passed = chain_a in chain_list
    assert if_passed


def test_vecs():
    pdb_path = TEST_PDB
    system = parse_pdb(pdb_path)
    model = system.models[1]
    all_atoms = AtomList()
    for chain in model:
        all_atoms += chain.atoms
    vecs = [atom.vec for atom in all_atoms]
    if_passed = all_atoms.vecs == vecs
    assert if_passed


def test_segids():
    pdb_path = TEST_PDB
    system = parse_pdb(pdb_path)
    atoms = system.atoms
    atoms.segids = atoms.chids
    segids = set([atom.segid for atom in atoms])
    chids = set(atoms.chids)
    if_passed = segids == chids
    assert if_passed


def test_sort():
    pdb_path = TEST_PDB
    system = parse_pdb(pdb_path)
    atoms = system.atoms
    randomized_atoms = AtomList(set(atoms))
    randomized_atoms.sort() # key=lambda atom: atom.atid
    if_passed = randomized_atoms == atoms
    assert if_passed
    

def test_filter():
    pdb_path = TEST_PDB
    chain_id_filter = ["A"]
    system = parse_pdb(pdb_path)
    atoms = system.atoms
    filtered_atoms = atoms.filter(chain_id_filter, mode="chid")
    chid = list(set(filtered_atoms.chids))[0]
    if_passed = chid == "A"
    assert if_passed


def test_remove():
    pdb_path = TEST_PDB
    mask = ["P", "C3'"]
    system = parse_pdb(pdb_path)
    atoms = system.atoms
    initial_len = len(atoms)
    filtered_atoms = atoms.filter(mask)
    atoms.remove_atoms(filtered_atoms)
    if_passed = initial_len - len(atoms) - len(filtered_atoms) == 0
    assert if_passed



if __name__ == "__main__":
    # test_read_pdb()     
    # test_save_system() 
    # test_save_atoms()
    # test_chain()  
    # test_vecs()   
    # test_segids()   
    test_sort()
    # test_filter()  
    # test_remove()  

  