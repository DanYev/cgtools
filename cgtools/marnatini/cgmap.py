import os
import sys
import copy
import time
import numpy as np

BEADS = ['BB1', 'BB2', 'BB3', 'SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SC6']
BY_ATOM_NAME = True # how to map residues


# Split each argument in a list                                               
def nsplit(*x):                                                               
    return [i.split() for i in x]  


def read_ndx_file(filename, by_indices=True):
    groups = {}
    with open(filename, 'r') as f:
        current_group = None
        for line in f:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                current_group = line[2:-2]
                groups[current_group] = []
            elif current_group:
                atoms = line.split()
                if by_indices:
                    indices = [int(x) - 1 for x in atoms]
                else:
                    indices = atoms
                groups[current_group].extend(indices)
    return groups
    

def get_mapping_byndx(new_or_old="new"):
    # Index files and mapping
    if new_or_old == "new":
        a_ndx = "A"
        c_ndx = "C"
        g_ndx = "G"
        u_ndx = "U"
    else:    
        a_ndx = "A_old"
        c_ndx = "C_old"
        g_ndx = "G_old"
        u_ndx = "U_old"
    a_mapping = read_ndx_file(f"ndx/{a_ndx}.ndx")
    c_mapping = read_ndx_file(f"ndx/{c_ndx}.ndx")
    g_mapping = read_ndx_file(f"ndx/{g_ndx}.ndx")
    u_mapping = read_ndx_file(f"ndx/{u_ndx}.ndx")
    mapping = {"A": a_mapping, "C": c_mapping, "G": g_mapping, "U": u_mapping}
    return mapping
    
    
def get_mapping_byname(new_or_old="new"):   
    if new_or_old == "new":
        BB_mapping = nsplit("P OP1 OP2 O5' O3' O1P O2P", 
                            "C5' 1H5' 2H5' C4' H4' O4' C3' H3'", 
                            "C1' C2' O2' O4'") # H1' 1H2' 2HO'
    else:
        BB_mapping = nsplit("P OP1 OP2 O5' O1P O2P O3'", 
                            "C4' O4' C5'", 
                            "C1' O2' C2' C3'")
                            
    if new_or_old == "new":
        mapping = {
            "A":  BB_mapping + nsplit(
                            "N9 C8 H8",
                            "N3 C4",
                            "N1 C2 H2",
                            "N6 C6 H61 H62",
                            "N7 C5"),
            "C":  BB_mapping + nsplit(
                            "N1 C5 C6",
                            "C2 O2",
                            "N3",
                            "N4 C4 H41 H42"),
            "G":  BB_mapping + nsplit(
                            "C8 H8 N9",
                            "C4 N3",
                            "C2 N2 H21 H22",
                            "N1", 
                            "C6 O6",
                            "C5 N7"),
            "U":  BB_mapping + nsplit(
                            "N1 C5 C6",
                            "C2 O2",
                            "N3 H3",
                            "C4 O4"),
        }

    mapping.update({"6MA":mapping["A"],
                    "2MA":mapping["A"],
                    "RA3":mapping["A"],
                    "RA5":mapping["A"],
                    "RAP":mapping["A"],                    
                    "DMA":mapping["A"],
                    "RC5":mapping["C"],
                    "5MC":mapping["C"],
                    "3MP":mapping["C"],
                    "MRC":mapping["C"],
                    "NMC":mapping["C"],
                    "RG5":mapping["G"],
                    "1MG":mapping["G"],
                    "2MG":mapping["G"],
                    "7MG":mapping["G"],
                    "MRG":mapping["G"],
                    "4SU":mapping["U"], 
                    "DHU":mapping["U"], 
                    "PSU":mapping["U"],
                    "5MU":mapping["U"],
                    "3MU":mapping["U"],
                    "MRU":mapping["U"],
                    "RU5":mapping["U"],
                    "RU3":mapping["U"]
    })
    return mapping
    
    
def map_residue(atom_dict, mapping, resname):
    """
    residue - CIF instance
    """
    res_mapping = mapping[resname]
    bvecs = np.zeros((len(res_mapping), 3))
    if not BY_ATOM_NAME:
        atom_vecs = atom_dict.values()
        for i, bead in enumerate(BEADS):
            if bead in res_mapping:
                indices = res_mapping[bead]
                vecs = np.take(atom_vecs, indices, 0)
                cog = np.average(vecs, axis=0)
                bvecs[i] = cog
    else:
        for i, bead in enumerate(BEADS):
            if i+1 <= len(res_mapping):
                names = res_mapping[i]
                vecs = [atom_dict[name] for name in names if name in atom_dict.keys()]
                cog = np.average(vecs, axis=0)
                bvecs[i] = cog
    return bvecs


def get_atom_dict(residue):
    atoms = residue.get_unpacked_list()
    atom_names = [atom.get_name() for atom in atoms]
    atom_vecs =  [atom.get_coord() for atom in atoms]
    atom_dict = dict(zip(atom_names, atom_vecs))
    return atom_dict
    
    
def update_atom_dict(resi, atom_dict, prev_dict, atom_names=("O3'", )):
    # Move O3' atom from the previous residue to the current residue.
    # resi is the current residue index
    for atom_name in atom_names:
        if atom_name in atom_dict.keys(): 
            if resi > 0:
                atom_dict.update({atom_name: prev_dict[atom_name]})
            else:
                atom_dict.pop(atom_name)


def main():
    print("DONE")

if __name__ == "__main__":
    main()

    
    

  
