import os
import sys
import copy
import time
import numpy as np

    
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
    n1 = np.cross(b1, b2); n1 /= np.linalg.norm(n1)
    n2 = np.cross(b2, b3); n2 /= np.linalg.norm(n2)
    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, b2n), n2)
    return np.degrees(np.arctan2(y, x))


def calc_bonds(model, bonds):
    """
    Calculates bond distances given top.bonds instance of Topology class,
    which contains connectivities, and an instance of Model class which contains a list of atoms
    Returns a BondList object
    """
    atoms = model.atoms()
    conns = bonds.connectivities
    params = bonds.parameters
    comms = bonds.comments
    pairs = [(atoms[i - 1], atoms[j - 1]) for i, j in conns]
    vecs_list = [(a1.vec, a2.vec) for a1, a2 in pairs]
    dists = [get_distance(*vecs) for vecs in vecs_list]
    resnames = [a1.resname for a1, a2 in pairs]
    params = [[param[0], metric] + param[2:] for param, metric in zip(params, dists)]
    comms = [f'{resname} {comm}' for comm, resname in zip(comms, resnames)]
    result = list(zip(conns, params, comms)) 
    return result


def calc_angles(model, angles): 
    """
    Calculates angles given top.angles instance of Topology class,
    which contains connectivities, and an instance of Model class which contains a list of atoms
    Returns a BondList object
    """
    atoms = model.atoms()
    conns = angles.connectivities
    params = angles.parameters
    comms = angles.comments
    triplets = [(atoms[i - 1], atoms[j - 1], atoms[k - 1]) for i, j, k in conns]
    vecs_list = [(a1.vec, a2.vec, a3.vec) for a1, a2, a3 in triplets]
    angles = [get_angle(*vecs) for vecs in vecs_list]
    resnames = [a1.resname for a1, a2, a3 in triplets]
    params = [[param[0], metric] + param[2:] for param, metric in zip(params, angles)]
    comms = [f'{resname} {comm}' for comm, resname in zip(comms, resnames)]
    result = list(zip(conns, params, comms)) 
    return result


def calc_dihedrals(model, dihs): 
    """
    Calculates dihedrals given top.dihs instance of Topology class,
    which contains connectivities, and an instance of Model class which contains a list of atoms
    Returns a BondList object
    """
    atoms = model.atoms()
    conns = dihs.connectivities
    params = dihs.parameters
    comms = dihs.comments
    quads = [(atoms[i - 1], atoms[j - 1], atoms[k - 1], atoms[l - 1]) for i, j, k, l in conns]
    vecs_list = [(a1.vec, a2.vec, a3.vec, a4.vec) for a1, a2, a3, a4 in quads]
    dihs = [get_dihedral(*vecs) for vecs in vecs_list]
    resnames = [a2.resname for a1, a2, a3, a4 in quads]
    params = [[param[0], metric] + param[2:] for param, metric in zip(params, dihs)]
    comms = [f'{resname} {comm}' for comm, resname in zip(comms, resnames)]
    result = list(zip(conns, params, comms)) 
    return result    
    
            
if __name__ == "__main__":
    pass
    

  
