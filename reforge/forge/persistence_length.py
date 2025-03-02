"""
Description:
    This module computes the persistence length of a nucleic acid system by
    calculating angles between consecutive bond vectors in a coarse-grained model.
    It provides functions to read a structure from a PDB file, compute geometric
    measures (angles, rotation matrices), and finally plot the average angles.

Author: DY
"""

import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser


def get_structure_cif(cif_id="path/to/cif"):
    """
    Read a structure from a CIF file.

    Args:
        cif_id (str): Path to the CIF file.

    Returns:
        structure: A Bio.PDB structure object.
    """
    parser = MMCIFParser()
    structure = parser.get_structure("structure", cif_id)
    return structure


def get_structure_pdb(pdb_id="path/to/pdb"):
    """
    Read a structure from a PDB file.

    Args:
        pdb_id (str): Path to the PDB file.

    Returns:
        structure: A Bio.PDB structure object.
    """
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_id)
    return structure


def get_residues(model):
    """
    Extract all residues from a model.

    Args:
        model: A Bio.PDB model object.

    Returns:
        list: A list of residue objects.
    """
    result = []
    for chain in model:
        for _, residue in enumerate(chain.get_unpacked_list()):
            result.append(residue)
    return result


def get_resid(residue):
    """
    Extract the residue id from the string representation of a residue.

    Args:
        residue: A Bio.PDB residue object.

    Returns:
        int: The residue id.
    """
    # Use the built-in repr function instead of __repr__()
    rep = repr(residue)
    try:
        # Expecting a format like "... resid=123 ..."
        return int(rep.split()[3].split("=")[1])
    except (IndexError, ValueError):
        raise ValueError("Could not extract residue id from: " + rep)


def get_atoms_by_name(residues, atom_name="BB3"):
    """
    Select atoms with a given name from a list of residues.

    Args:
        residues (list): List of residue objects.
        atom_name (str): Name of the atom to select (default "BB3").

    Returns:
        tuple: A tuple (atoms, resids) where atoms is a list of matching atom objects,
               and resids is a list of corresponding residue ids.
    """
    atoms = []
    resids = []
    for residue in residues:
        rid = get_resid(residue)
        for atom in residue.get_atoms():
            if atom.get_name() == atom_name:
                atoms.append(atom)
                resids.append(rid)
    return atoms, resids


def get_coords(atoms):
    """
    Get the coordinates for a list of atoms.

    Args:
        atoms (list): List of atom objects.

    Returns:
        list: List of coordinate arrays.
    """
    return [atom.get_coord() for atom in atoms]


def get_angle(v1, v2):
    """
    Compute the angle (in degrees) between two vectors.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        float: Angle in degrees.
    """
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2.

    Args:
        vec1 (array-like): Source vector.
        vec2 (array-like): Destination vector.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    if (vec1 == vec2).all():
        return np.eye(3)
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def persistence_length(structure):
    """
    Calculate and plot the persistence length from a structure.

    For each model in the structure, the function computes the angle between
    bond vectors (taken from atoms with name "BB3") after aligning them with
    a reference vector. The average angles are then plotted and saved to a file.

    Args:
        structure: A Bio.PDB structure object.
    """
    all_angles = []
    global_vec0 = None
    global_rmats = None
    for idx, model in enumerate(structure):
        residues = get_residues(model)
        atoms, _ = get_atoms_by_name(residues, atom_name="BB3")
        coords = np.array(get_coords(atoms))
        if len(coords) < 11:
            continue  # Not enough data
        vecs = coords[1:] - coords[:-1]
        # For the first model, set the reference vector and rotation matrices.
        if idx == 0:
            rvec = vecs[10]
            global_vec0 = vecs[0]
            global_rmats = [rotation_matrix_from_vectors(vec, rvec) for vec in vecs]
        # Use global values from the first model
        rvec = vecs[10]
        # Apply rotation matrices to align bond vectors.
        vecs_trans = np.einsum("ijk,ik->ij", global_rmats, vecs)
        # Compute angles between each transformed vector and the reference vector.
        angles = [get_angle(vecs_trans[i], rvec) for i in range(len(vecs_trans))]
        all_angles.append(angles)
    if not all_angles:
        print("No valid angle data found.", file=plt.sys.stderr)
        return
    all_angles = np.array(all_angles)
    av_angles = np.average(all_angles, axis=0)
    print(av_angles)
    fig = plt.figure(figsize=(16.0, 6.0))
    plt.plot(np.arange(len(av_angles))[:200], np.abs(av_angles)[:200])
    fig.savefig("pers_length.png")
    plt.close()


def test_rmats():
    """
    Test function for rotation_matrix_from_vectors.
    """
    vecs = np.array([[1, 2, 3], [3, 2, 1], [4, 2, 6]])
    rmats = [rotation_matrix_from_vectors(vec, vecs[2]) for vec in vecs]
    print(np.einsum("ijk,ik->ij", rmats, vecs))


def main():
    """
    Main function to compute persistence length.
    """
    wdir = "systems/100bpRNA/mdrun"
    structure = get_structure_pdb(f"{wdir}/md.pdb")
    persistence_length(structure)


if __name__ == "__main__":
    main()
