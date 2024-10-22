import os
import sys
import copy
import time
import numpy as np
import itpio

    
def get_distance(v1, v2):
    return np.linalg.norm(v1 - v2) / 10.0  
    

def get_angle(vec1, vec2, vec3):
    v1 = vec1 - vec2
    v2 = vec3 - vec2
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees    


def get_dihedral(v1, v2, v3, v4, flip_sign=False):
    b1 = v2 - v1
    b2 = v3 - v2
    b3 = v4 - v3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    angle = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))
    sign = np.dot(np.cross(n1, n2), b2)
    if flip_sign and sign < 0:
        angle = -angle
    if not flip_sign and sign < 0:
        angle = -angle + 2 * np.pi 
    angle_degrees = np.degrees(angle)
    return angle_degrees - 180
    
            
def get_bonds(vecs, itp_data):
    bonds = []
    pairs = itp_data['bonds'].keys()
    for pair in pairs:
        i, j = pair
        bond = get_distance(vecs[i], vecs[j])
        bonds.append(bond)
    return bonds
    
    
def get_angles(vecs, itp_data):    
    angles = []
    triplets = itp_data['angles'].keys()
    for triplet in triplets:
        i, j, k = triplet
        angle = get_angle(vecs[i], vecs[j], vecs[k])
        angles.append(angle)
    return angles


def get_dihedrals(vecs, itp_data):
    dihedrals = []
    quads = itp_data['dihedrals'].keys()
    for quad in quads:
        i, j, k, l = quad
        dihedral = get_dihedral(vecs[i], vecs[j], vecs[k], vecs[l])
        dihedrals.append(dihedral)
    return dihedrals


if __name__ == "__main__":
    pass
    

  
