from cgtools.pdbtools import System, PDBParser
import numpy as np
import copy


def read_pdb(pdb_path):
    parser = PDBParser(pdb_path)
    system = parser.parse()
    print(system)  # System summary
    return system


def move_o3(system):
    for i, residue in enumerate(system.residues()):
        atoms = residue.atoms()
        for atom in atoms:
            if atom.name == "O3'":
                atoms.remove(atom)
                if i == 0:
                    o3atom = atom
                    break
                else:
                    o3atom.resname = residue.resname  
                    o3atom.resid = residue.resid   
                    atoms.append(o3atom)  
                    o3atom = atom 
                    break


def map_residue(residue, mapping, atid):
    cgresidue = []
    dummy_atom = residue.atoms()[0]
    for bname, anames in mapping.items():
        bead = copy.deepcopy(dummy_atom)
        bead.name = bname
        bead.serial = atid
        atid += 1
        atoms = [atom for atom in residue.atoms() if atom.name in anames]
        vecs = [(atom.x, atom.y, atom.z) for atom in atoms]
        bvec = np.average(vecs, axis=0)
        bead.x = bvec[0]
        bead.y = bvec[1]
        bead.z = bvec[2]
        if bname.startswith('B'):
            bead.element = 'Z'
        else:
            bead.element = 'S'
        cgresidue.append(bead)
    return cgresidue


def map_residues(system, ff, atid=1):
    cgchain = []
    for idx, residue in enumerate(system.residues()):
        mapping = ff.mapping[residue.resname]
        if idx == 0:
            del mapping["BB1"]
        cgresidue = map_residue(residue, mapping, atid)
        cgchain.extend(cgresidue)
        atid += len(mapping)   
    return cgchain


def save_pdb(atoms, fpath='test.pdb'):
    system = System()
    system.add_atoms(atoms, model_id=1)
    system.save_pdb(fpath)