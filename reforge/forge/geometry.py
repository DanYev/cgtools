import os
import sys
import copy
import time
import numpy as np
import reforge.forge.cgmap as cgmap
from reforge.forge.topology import Topology, BondList


def get_distance(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.linalg.norm(v1 - v2) / 10.0


def get_angle(v1, v2, v3):
    v1, v2, v3 = map(np.array, (v1, v2, v3))
    v1 = v1 - v2
    v2 = v3 - v2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


def get_dihedral(v1, v2, v3, v4):
    v1, v2, v3, v4 = map(np.array, (v1, v2, v3, v4))
    b1, b2, b3 = v2 - v1, v3 - v2, v4 - v3
    b2n = b2 / np.linalg.norm(b2)
    n1 = np.cross(b1, b2)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(b2, b3)
    n2 /= np.linalg.norm(n2)
    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, b2n), n2)
    return np.degrees(np.arctan2(y, x))


def calc_bonds(atoms, bonds):
    """Calculates bond distances given top.bonds instance of Topology class,
    which contains connectivities, and a list of pdb Atom objects Returns a
    BondList object."""
    conns = bonds.conns
    params = bonds.params
    comms = bonds.comms
    pairs = [(atoms[i - 1], atoms[j - 1]) for i, j in conns]
    vecs_list = [(a1.vec, a2.vec) for a1, a2 in pairs]
    dists = [get_distance(*vecs) for vecs in vecs_list]
    resnames = [a1.resname for a1, a2 in pairs]
    params = [[param[0], metric] + param[2:] for param, metric in zip(params, dists)]
    comms = [f"{resname} {comm}" for comm, resname in zip(comms, resnames)]
    result = list(zip(conns, params, comms))
    return BondList(result)


def calc_angles(atoms, angles):
    """Calculates angles given top.angles instance of Topology class, which
    contains connectivities, and an instance of Model class which contains a
    list of atoms Returns a BondList object."""
    conns = angles.conns
    params = angles.params
    comms = angles.comms
    triplets = [(atoms[i - 1], atoms[j - 1], atoms[k - 1]) for i, j, k in conns]
    vecs_list = [(a1.vec, a2.vec, a3.vec) for a1, a2, a3 in triplets]
    angles = [get_angle(*vecs) for vecs in vecs_list]
    resnames = [a1.resname for a1, a2, a3 in triplets]
    params = [[param[0], metric] + param[2:] for param, metric in zip(params, angles)]
    comms = [f"{resname} {comm}" for comm, resname in zip(comms, resnames)]
    result = list(zip(conns, params, comms))
    return BondList(result)


def calc_dihedrals(atoms, dihs):
    """Calculates dihedrals given top.dihs instance of Topology class, which
    contains connectivities, and an instance of Model class which contains a
    list of atoms Returns a BondList object."""
    conns = dihs.conns
    params = dihs.params
    comms = dihs.comms
    quads = [
        (atoms[i - 1], atoms[j - 1], atoms[k - 1], atoms[l - 1]) for i, j, k, l in conns
    ]
    vecs_list = [(a1.vec, a2.vec, a3.vec, a4.vec) for a1, a2, a3, a4 in quads]
    dihs = [get_dihedral(*vecs) for vecs in vecs_list]
    resnames = [a2.resname for a1, a2, a3, a4 in quads]
    params = [[param[0], metric] + param[2:] for param, metric in zip(params, dihs)]
    comms = [f"{resname} {comm}" for comm, resname in zip(comms, resnames)]
    result = list(zip(conns, params, comms))
    return BondList(result)


def get_cg_bonds(inpdb, top):
    """
    Calculates bonds, angles, dihedrals given a CG system .pdb and the reference topology oblect
    Returns three BondList objects: bonds, angles, dihedrals
    """
    print(f"Calculating bonds, angles and dihedrals from {inpdb}...", file=sys.stderr)
    system = cgmap.read_pdb(inpdb)
    bonds, angles, dihs = BondList(), BondList(), BondList()
    for model in system:
        bonds.extend(calc_bonds(model.atoms(), top.bonds + top.cons))
        angles.extend(calc_angles(model.atoms(), top.angles))
        dihs.extend(calc_dihedrals(model.atoms(), top.dihs))
    print("Done!", file=sys.stderr)
    return bonds, angles, dihs


def get_aa_bonds(inpdb, ff, top):
    """
    Calculates bonds, angles, dihedrals given an AA system .pdb and the reference topology oblect
    Returns three BondList objects: bonds, angles, dihedrals
    """
    print(f"Calculating bonds, angles and dihedrals from {inpdb}...", file=sys.stderr)
    system = cgmap.read_pdb(inpdb)
    cgmap.move_o3(system)
    bonds, angles, dihs = BondList(), BondList(), BondList()
    for model in system:
        model = cgmap.map_model(model, ff, atid=1, merge=True)
        bonds.extend(calc_bonds(model, top.bonds + top.cons))
        angles.extend(calc_angles(model, top.angles))
        dihs.extend(calc_dihedrals(model, top.dihs))
    print("Done!", file=sys.stderr)
    return bonds, angles, dihs


if __name__ == "__main__":
    pass
