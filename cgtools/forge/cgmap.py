from cgtools.pdbtools import System, PDBParser
import numpy as np
import copy

def read_pdb(pdb_path):
    """
    Read a PDB file and parse it into a system object.

    Parameters:
        pdb_path (str): The file path to the PDB file.

    Returns:
        System: A system object representing the parsed PDB structure.
    """
    parser = PDBParser(pdb_path)
    system = parser.parse()
    return system


def move_o3(system):
    """
    Move each O3' atom to the next residue.
    Needed for some CG Nucleic FFs due to the phosphate group mapping
   
    Parameters:
        system: A system object containing chains and residues.
    """
    for chain in system.chains():
        for i, residue in enumerate(chain):
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
    """
    Map an atomistic residue to a coarse-grained (CG) residue.

    For each bead defined in the mapping dictionary, this function creates a new bead atom.
    The bead's coordinates are determined by averaging the coordinates of the atoms in the
    original residue that match the bead's atom names.

    Parameters:
        residue: The original residue object.
        mapping (dict): A dictionary where keys are bead names and values are lists of atom 
                        names to be included in that bead.
        atid (int): The starting atom id to assign to new beads.

    Returns:
        list: A list of new bead atoms representing the coarse-grained residue.
    """
    cgresidue = []
    dummy_atom = residue.atoms()[0]
    for bname, anames in mapping.items():
        bead = copy.deepcopy(dummy_atom)
        bead.name = bname
        bead.atid = atid
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


def map_residues(chain, ff, atid=1):
    """
    Map a chain of atomistic residues to a coarse-grained (CG) representation.

    For each residue in the chain, the function retrieves the corresponding bead mapping 
    from the force field (ff) based on the residue's name. For the first residue in the 
    chain, the mapping for "BB1" is removed. Each residue is then converted to its CG 
    representation using the map_residue function, and the resulting beads are collected 
    into a single list.

    Parameters:
        chain: A list of residue objects.
        ff: A force field object that contains a 'mapping' dictionary keyed by residue name.
        atid (int): The starting atom id for the new beads (default is 1).

    Returns:
        list: A list of coarse-grained bead atoms representing the entire chain.
    """
    cgchain = []
    for idx, residue in enumerate(chain):
        mapping = ff.mapping[residue.resname]
        if idx == 0:
            mapping = mapping.copy()
            del mapping["BB1"]     
        cgresidue = map_residue(residue, mapping, atid)
        cgchain.extend(cgresidue)
        atid += len(mapping)   
    return cgchain


def save_pdb(atoms, fpath='test.pdb'):
    """
    Save a list of atoms to a PDB file.

    This function creates a new System object, adds the provided atoms to it under model_id 1,
    and saves the system to a PDB file at the specified path.

    Parameters:
        atoms (list): A list of atom objects to be saved.
        fpath (str): The file path where the PDB file will be saved (default is 'test.pdb').
    """
    system = System()
    system.add_atoms(atoms, model_id=1)
    system.save_pdb(fpath)
