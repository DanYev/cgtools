import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import vectors, Superimposer, PDBIO, Atom

    
def get_structure_cif(cif_id="path/to/cif"):
    # BIO.structure object
    parser = MMCIFParser()
    structure = parser.get_structure("structure", cif_id)
    return structure
    

def get_structure_pdb(pdb_id="path/to/pdb"):
    # BIO.structure object
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_id)
    return structure


def get_residues(model):
    result = []
    for chain in model:
        residues = chain.get_unpacked_list()
        for resi, residue in enumerate(residues):
            result.append(residue)
    return result
    
    
def get_resid(residue):
    return int(residue.__repr__().split()[3].split('=')[1])
    
    
def get_atoms_by_name(residues, atom_name='BB3'):
    atoms = []
    resids = []
    for residue in residues:
        resid = get_resid(residue)
        resatoms = residue.get_atoms()
        for atom in resatoms:
            name = atom.get_name()
            if name == atom_name:
                atoms.append(atom)
                resids.append(resid)
    return atoms, resids
    

def get_coords(atoms):
    return [atom.get_coord() for atom in atoms]
    

def get_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)
    return cos_theta    
    
    
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    if (vec1 == vec2).all():
        rotation_matrix = np.eye(3)
    else:
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
    

def persistence_length(structure):
    all_angles = []
    for idx, model in enumerate(structure):
        residues = get_residues(model)
        atoms, resids = get_atoms_by_name(residues, atom_name='BB3')
        coords = np.array(get_coords(atoms))
        vecs = coords[1:] - coords[:-1]
        if idx == 0:
            rvec = vecs[10]
            vec0 = vecs[0]
            rmats = [rotation_matrix_from_vectors(vec, rvec) for vec in vecs]
        rvec = vecs[10]
        rmat2 = rotation_matrix_from_vectors(rvec, vec0)
        vecs = np.einsum('ijk,ik->ij', rmats, vecs)
        # vecs = np.einsum('jk,ik->ij', rmat2, vecs)
        angles = []
        for i in range(len(vecs)):
            angle = get_angle(vecs[i], rvec)
            angles.append(angle)
        all_angles.append(angles)
    all_angles = np.array(all_angles)
    av_angles = np.average(all_angles, axis=0)
    print(av_angles)
    fig = plt.figure(figsize=(16.0, 6.0))
    plt.plot(np.arange(len(av_angles))[0:200], np.abs(av_angles)[0:200])
    fig.savefig(f'pers_length.png')
    plt.close()
    
    
def test_rmats():
    vecs = np.array([[1,2,3], [3,2,1], [4,2,6]])
    rmats = [rotation_matrix_from_vectors(vec, vecs[2]) for vec in vecs]
    print(np.einsum('ijk,ik->ij', rmats, vecs))

    
def main():
    wdir = 'systems/100bpRNA/mdrun'
    # wdir = 'cif'
    structure = get_structure_pdb(f"{wdir}/md.pdb")
    persistence_length(structure)


if __name__ == "__main__":
    main()

    
    

  
