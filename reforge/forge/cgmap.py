"""CG mapping tools

Description:
    Provides functions to map atomistic PDB data to a coarse-grained (CG)
    representation.

Author: DY
"""

import copy
import numpy as np

def move_o3(system):
    """Move each O3' atom to the next residue. Needed for some CG nucleic
    force fields due to the phosphate group mapping.

    Parameters:
        system: A system object containing chains and residues.
    """
    for chain in system.chains():
        for i, residue in enumerate(chain):
            atoms = residue.atoms
            for atom in atoms:
                if atom.name == "O3'":
                    atoms.remove(atom)
                    if i == 0:
                        o3atom = atom
                    else:
                        o3atom.resname = residue.resname
                        o3atom.resid = residue.resid
                        atoms.append(o3atom)
                        o3atom = atom
                    break


def map_residue(residue, mapping, atid):
    """Map an atomistic residue to a coarse-grained (CG) residue.

    For each bead defined in the mapping dictionary, this function creates a new bead atom.
    The bead's coordinates are determined by averaging the coordinates of the atoms in the
    original residue that match the bead's atom names.

    Parameters:
        residue: The original residue object.
        mapping (dict): Dictionary with bead names as keys and lists of atom names as values.
        atid (int): Starting atom id to assign to new beads.

    Returns:
        list: List of new bead atoms representing the coarse-grained residue.
    """
    cgresidue = []
    dummy_atom = residue.atoms[0]
    for bname, anames in mapping.items():
        bead = copy.deepcopy(dummy_atom)
        bead.name = bname
        bead.atid = atid
        atid += 1
        atoms = residue.atoms.mask(anames)
        bvec = np.average(atoms.vecs, axis=0)
        bead.x, bead.y, bead.z = bvec[0], bvec[1], bvec[2]
        bead.vec = bvec
        bead.element = "Z" if bname.startswith("B") else "S"
        cgresidue.append(bead)
    return cgresidue


def map_chain(chain, ff, atid=1):
    """Map a chain of atomistic residues to a coarse-grained (CG) representation.

    For each residue in the chain, the function retrieves the corresponding bead mapping
    from the force field (ff) based on the residue's name. For the first residue, the mapping
    for "BB1" is removed. Each residue is then converted to its CG representation using the
    map_residue function, and the resulting beads are collected into a single list.

    Parameters:
        chain: List of residue objects.
        ff: Force field object that contains a 'mapping' dictionary keyed by residue name.
        atid (int): Starting atom id for new beads (default is 1).

    Returns:
        list: List of coarse-grained bead atoms representing the entire chain.
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


def map_model(model, ff, atid=1):
    """
    Map a model of atomistic residues to a coarse-grained (CG) representation.

    Parameters:
        model: Iterable of chains (each a list of residue objects).
        ff: Force field object containing a 'mapping' dictionary keyed by residue name.
        atid (int): Starting atom id for new beads (default is 1).

    Returns:
        list: List of coarse-grained bead atoms representing the entire model.
    """
    cgmodel = []
    for chain in model:
        cgchain = map_chain(chain, ff, atid)
        cgmodel.extend(cgchain)
        atid += len(cgchain)
    return cgmodel
