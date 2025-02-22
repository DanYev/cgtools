import os 
from cgtools.pdbtools import Atom, AtomList, System, Model, Chain, Residue, parse_pdb, sort_pdb

TEST_PDB = "in_test.pdb" 

def test_read_pdb():
    in_pdb = TEST_PDB
    system = parse_pdb(in_pdb)
    atoms = system.atoms
    if_passed = len(atoms) > 0
    assert if_passed


def test_save_system():
    in_pdb = TEST_PDB 
    test_pdb = "test.pdb"
    if test_pdb in os.listdir():
        os.remove(test_pdb)
    system = parse_pdb(in_pdb)
    system.write_to_pdb(test_pdb)
    if_passed = test_pdb in os.listdir()
    assert if_passed


def test_save_atoms():
    in_pdb = TEST_PDB
    test_pdb = "test.pdb"
    if test_pdb in os.listdir():
        os.remove(test_pdb)
    system = parse_pdb(in_pdb)
    atoms = system.atoms
    atoms.write_to_pdb(test_pdb)
    if_passed = test_pdb in os.listdir()
    assert if_passed


def test_chain():
    in_pdb = TEST_PDB
    system = parse_pdb(in_pdb)
    model = system.models[1]
    chids = list(model.chains.keys())
    chain = model.chains[chids[0]]
    chain_list = model.select_chains(chids)
    if_passed = chain in chain_list
    assert if_passed


def test_vecs():
    in_pdb = TEST_PDB
    system = parse_pdb(in_pdb)
    model = system.models[1]
    all_atoms = AtomList()
    for chain in model:
        all_atoms += chain.atoms
    vecs = [atom.vec for atom in all_atoms]
    if_passed = all_atoms.vecs == vecs
    assert if_passed


def test_segids():
    in_pdb = TEST_PDB
    system = parse_pdb(in_pdb)
    atoms = system.atoms
    atoms.segids = atoms.chids
    segids = set([atom.segid for atom in atoms])
    chids = set(atoms.chids)
    if_passed = segids == chids
    assert if_passed


def test_sort():
    in_pdb = TEST_PDB
    system = parse_pdb(in_pdb)
    atoms = system.atoms
    randomized_atoms = atoms # AtomList(set(atoms))
    randomized_atoms.sort() # key=lambda atom: atom.atid
    if_passed = randomized_atoms == atoms
    assert if_passed
    

def test_filter():
    in_pdb = TEST_PDB
    system = parse_pdb(in_pdb)
    atoms = system.atoms
    chids = list(set(atoms.chids))
    chid = chids[0]
    filtered_atoms = atoms.filter(chid, mode="chid")
    test_chid = list(set(filtered_atoms.chids))[0]
    if_passed = chid == test_chid
    assert if_passed


def test_remove():
    in_pdb = TEST_PDB
    mask = ["P", "C3'"]
    system = parse_pdb(in_pdb)
    atoms = system.atoms
    initial_len = len(atoms)
    filtered_atoms = atoms.filter(mask)
    atoms.remove_atoms(filtered_atoms)
    if_passed = initial_len - len(atoms) - len(filtered_atoms) == 0
    assert if_passed


def test_sort_pdb():
    in_pdb = TEST_PDB
    out_pdb = 'test.pdb'
    sort_pdb(in_pdb, out_pdb)
    if_passed = True
    assert if_passed


def test_write_ndx():
    in_pdb = TEST_PDB
    out_ndx = 'test.ndx'
    if out_ndx in os.listdir():
        os.remove(out_ndx)
    system = parse_pdb(in_pdb)
    atoms = system.atoms
    atoms.write_to_ndx(out_ndx, header=f'[ System ]', append=False, wrap=15) # sys ndx
    if_passed = out_ndx in os.listdir()
    os.remove(out_ndx)
    assert if_passed
    

if __name__ == "__main__":
    test_read_pdb()     
    test_save_system() 
    test_save_atoms()
    test_chain()  
    test_vecs()   
    test_segids()   
    test_sort()
    test_filter()  
    test_remove()  
    test_sort_pdb()
    test_write_ndx()

  