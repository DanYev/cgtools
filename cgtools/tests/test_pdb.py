import os 
from cgtools.pdbtools import PDBParser, Atom, AtomList, System, Model, Chain, Residue, parse_pdb


def test_read_pdb():
    pdb_path = "dsRNA.pdb"  
    parser = PDBParser(pdb_path)
    system = parser.parse()
    atoms = system.atoms
    print(f"\nTotal atoms parsed: {len(atoms)}")


def test_save_system():
    pdb_path = "dsRNA.pdb" 
    test_pdb = "test.pdb"
    if test_pdb in os.listdir():
        os.remove(test_pdb)
    system = parse_pdb(pdb_path)
    system.save_pdb(test_pdb)
    if_passed = test_pdb in os.listdir()
    assert if_passed


def test_save_atoms():
    pdb_path = "dsRNA.pdb" 
    test_pdb = "test.pdb"
    if test_pdb in os.listdir():
        os.remove(test_pdb)
    system = parse_pdb(pdb_path)
    atoms = system.atoms
    atoms.save_pdb(test_pdb)
    if_passed = test_pdb in os.listdir()
    assert if_passed


def test_filter():
    chain_id_filter = ["A"]
    pdb_path = "dsRNA.pdb" 
    system = parse_pdb(pdb_path)
    atoms = system.atoms
    masked_atoms = atoms.filter(chain_id_filter, mode="chid")
    chid = list(set(masked_atoms.chids))[0]
    if_passed = chid == "A"
    assert if_passed


def test_seg_id():
    name_filter = ["P", "C3'"]
    chain_id_filter = ["A"]
    pdb_path = "dsRNA.pdb" 
    system = parse_pdb(pdb_path)
    atoms = system.atoms
    masked_atoms = atoms.filter(name_filter, mode="name")
    masked_atoms = atoms.filter(chain_id_filter, mode="chid")
    print(masked_atoms)
    exit()
    if_passed = initial_len - len(atoms) - len(masked_atoms) == 0
    assert if_passed


def test_remove():
    mask = ["P", "C3'"]
    pdb_path = "dsRNA.pdb" 
    system = parse_pdb(pdb_path)
    atoms = system.atoms
    initial_len = len(atoms)
    filtered_atoms = atoms.filter(mask)
    atoms.remove_atoms(filtered_atoms)
    if_passed = initial_len - len(atoms) - len(filtered_atoms) == 0
    assert if_passed


def test_chain():
    pdb_path = "dsRNA.pdb" 
    system = parse_pdb(pdb_path)
    model = system.models[1]
    chain_a = model.chains["A"]
    chain_list = model.select_chains(["A", "B"])
    if_passed = chain_a in chain_list
    assert if_passed


def test_vecs():
    pdb_path = "dsRNA.pdb" 
    system = parse_pdb(pdb_path)
    model = system.models[1]
    all_atoms = AtomList()
    for chain in model:
        all_atoms += chain.atoms
    vecs = [atom.vec for atom in all_atoms]
    if_passed = all_atoms.vecs == vecs
    assert if_passed

    

if __name__ == "__main__":
    test_read_pdb()     
    test_save_system() 
    test_save_atoms()
    test_filter()  
    test_remove()     
    test_chain()  
    test_vecs()       