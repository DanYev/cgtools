import os
import time
import sys
import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import vectors, Superimposer, PDBIO, Atom


# molecule = sys.argv[1]
# mode = sys.argv[2]


sys.path.append('./cgtools')
import itpio
import marnatini.cgmap as cgmap
from marnatini.geometry import get_angle, get_bonds, get_angles, get_dihedrals
from marnatini.plot_geometry import plot_geometry, make_fig_bonded, save_data

res_dict = {'ADE': 'A', 'CYT': 'C', 'GUA': 'G', 'URA': 'U'}

# Split each argument in a list                                               
def nsplit(*x):                                                               
    return [i.split() for i in x]  
    
    
def make_structure_cif(cif_id="path/to/cif"):
    # BIO.structure object
    parser = MMCIFParser()
    structure = parser.get_structure("structure", cif_id)
    return structure
    

def make_structure_pdb(pdb_id="path/to/pdb"):
    # BIO.structure object
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_id)
    return structure
    
    
def read_topology(  a_itp='itp/nucleotides/adenine.itp', 
                    c_itp='itp/nucleotides/cytosine.itp', 
                    g_itp='itp/nucleotides/guanine.itp', 
                    u_itp='itp/nucleotides/uracil.itp'):
    a_top = itpio.read_itp(a_itp)
    u_top = itpio.read_itp(u_itp)
    c_top = itpio.read_itp(c_itp)
    g_top = itpio.read_itp(g_itp)
    topology = {'A': a_top, 'C': c_top, 'G': g_top, 'U': u_top}
    return topology
    
    
def get_geometry(vecs, topology):
    bonds = get_bonds(vecs, topology)
    angles = get_angles(vecs, topology)
    dihedrals = get_dihedrals(vecs, topology)
    return bonds, angles, dihedrals
    

def shizz_to_get_done(all_bonds, all_angles, all_dihedrals, atom_dict, resname, next_dict, next_resname, prev_dict, prev_resname, mapping, topology):
    if mapping:
        vecs = cgmap.map_residue(atom_dict, mapping, resname)
        bb_nres_vecs = cgmap.map_residue(next_dict, mapping, next_resname)[0:4]
        bb_pres_vecs = cgmap.map_residue(prev_dict, mapping, prev_resname)[0:4]
    else:
        vecs = [value for value in atom_dict.values()]
        bb_nres_vecs = [value for value in next_dict.values()][0:4]
        bb_pres_vecs = [value for value in prev_dict.values()][0:4]
    # vecs.extend(bb_nres_vecs).append(bb_pres_vecs[1])
    vecs = np.append(vecs, bb_nres_vecs, 0)
    vecs = np.append(vecs, [bb_pres_vecs[1]], 0)
    bonds, angles, dihedrals = get_geometry(vecs, topology)
    all_bonds.append(bonds)
    all_angles.append(angles)
    all_dihedrals.append(dihedrals)
    
    
def iterate_and_get_shizz_done(structure, mapping, resnames, topology):
    bonds = []
    angles = []
    dihedrals = []
    for model in structure:
        for chain in model:
            residues = chain.get_unpacked_list()
            for resi, residue in enumerate(residues):
                if resi <= 1 or resi >= len(residues)-1:
                    continue
                resname = residue.get_resname()
                atom_dict = cgmap.get_atom_dict(residue)
                prev_resname = residues[resi-1].get_resname()
                prev_dict = cgmap.get_atom_dict(residues[resi-1])
                next_resname = residues[resi+1].get_resname()
                next_dict = cgmap.get_atom_dict(residues[resi+1])
                cgmap.update_atom_dict(resi+1, next_dict, atom_dict, ("O3'", ))
                cgmap.update_atom_dict(resi, atom_dict, prev_dict, ("O3'", ))
                if resname in resnames:
                    shizz_to_get_done(bonds, angles, dihedrals, atom_dict, resname, next_dict, next_resname, prev_dict, prev_resname, mapping, topology[resname])
    return bonds, angles, dihedrals
    
    
def iterate_and_get_shizz_done_aa(structure, mapping, resnames, topology):
    bonds = []
    angles = []
    dihedrals = []
    for model in structure:
        for chain in model:
            residues = chain.get_unpacked_list()
            for resi, residue in enumerate(residues):
                if resi <= 1 or resi >= len(residues)-1:
                    continue
                resname = residue.get_resname()
                resname = res_dict[resname]
                atom_dict = cgmap.get_atom_dict(residue)
                prev_resname = residues[resi-1].get_resname()
                prev_resname = res_dict[prev_resname]
                prev_dict = cgmap.get_atom_dict(residues[resi-1])
                next_resname = residues[resi+1].get_resname()
                next_resname = res_dict[next_resname]
                next_dict = cgmap.get_atom_dict(residues[resi+1])
                cgmap.update_atom_dict(resi+1, next_dict, atom_dict, ("O3'", ))
                cgmap.update_atom_dict(resi, atom_dict, prev_dict, ("O3'", ))
                if resname in resnames:
                    shizz_to_get_done(bonds, angles, dihedrals, atom_dict, resname, next_dict, next_resname, prev_dict, prev_resname, mapping, topology[resname])
    return bonds, angles, dihedrals


def pdb_iterator(structure):
    separator = '-' * 76
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(separator)
            print(f'Calling function: {func.__name__}')
            start_time = time.time()
            result = []
            for model in structure:
                for chain in model:
                    residues = chain.get_unpacked_list()
                    for resi, residue in enumerate(residues):
                        kwargs['residue'] = residue
                        f = func(*args, **kwargs)
                        result.append(f)
            end_time = time.time()
            print(f'Function {func.__name__} completed')
            print(f'Time taken: {end_time - start_time:.4f} seconds')
            print(separator)
            return result
        return wrapper
    return decorator
    
    
def get_average(bond):
    return np.average(np.array(bond).T, axis=-1)


def get_std(bond):
    return np.std(np.array(bond).T, axis=-1)
    
    
def update_topology(topology, bonds, angles, dihedrals, std_bonds, std_angles, std_dihedrals):
    fcs = [20000, 200, 20]
    ftypes = [1, 1, 1]
    sections = ['bonds', 'angles', 'dihedrals']
    glist = [bonds, angles, dihedrals]
    stdlist = [std_bonds, std_angles, std_dihedrals]
    for section, bonds, stds, fc, ftype in zip(sections, glist, stdlist, fcs, ftypes):
        for key, bond, std in zip(topology[section].keys(), bonds, stds):
            if topology[section][key][2] != 1000000.0:
                if section == 'dihedrals':
                    params = (topology[section][key][0], f"{bond:.3f}", topology[section][key][2], int(topology[section][key][3]),  f";{std:.3f}")
                else:
                    params = (topology[section][key][0], f"{bond:.3f}", topology[section][key][2], f";{std:.3f}")
                # params = (topology[section][key][0], bond, fc)
                topology[section][key] = params
                
                
def bonded_parameters():
    resnames = ('A', 'C', 'G', 'U') # ('A', 'C', 'G', 'U')  ('A', 'U') 
    version = 'new'
    molecule = "RNA"

    
    # system = sys.argv[1]
    # mdrun = sys.argv[2] 
    topology = read_topology(   a_itp=f'cgtools/itp/nucbonded/plot_A_{version}.itp', 
                                c_itp=f'cgtools/itp/nucbonded/plot_C_{version}.itp', 
                                g_itp=f'cgtools/itp/nucbonded/plot_G_{version}.itp', 
                                u_itp=f'cgtools/itp/nucbonded/plot_U_{version}.itp')
                                
    # if mode == 'aa':
    #     mapping = cgmap.get_mapping_byname(version)
    # if mode == 'cg':
    #     mapping = None
    
    # AA structure
    aa_structure = make_structure_pdb(f"systems/dsRNA_aa/mdruns/mdrun_2/mdc.pdb")
    # aa_structure = make_structure_pdb(f"ribosomes_old/test.pdb")
    cg_structure = make_structure_pdb(f"systems/dsRNA/mdruns/mdrun_2/mdc.pdb")
    # cg_structure = make_structure_pdb(f"systems/{system}/cgpdb/chain_A.pdb")
    mapping = cgmap.get_mapping_byname('new')
    all_aa_bonds, all_aa_angles, all_aa_dihedrals, all_cg_bonds, all_cg_angles, all_cg_dihedrals = [], [], [], [], [], []
    for resname in resnames:
        name = f"{molecule}_{resname}_{version}"
        bonds, angles, dihedrals = iterate_and_get_shizz_done_aa(aa_structure, mapping, resname, topology)
        cgbonds, cgangles, cgdihedrals = iterate_and_get_shizz_done(cg_structure, None, resname, topology)
        all_aa_bonds.extend(bonds)
        all_aa_angles.extend(angles)
        all_aa_dihedrals.extend(dihedrals)
        all_cg_bonds.extend(cgbonds)
        all_cg_angles.extend(cgangles)
        all_cg_dihedrals.extend(cgdihedrals)
        plot_geometry(bonds, angles, dihedrals, cgbonds, cgangles, cgdihedrals, topology[resname], molecule, resname)
        # av_bonds, av_angles, av_dihedrals = get_average(bonds), get_average(angles), get_average(dihedrals)
        # std_bonds, std_angles, std_dihedrals = get_std(bonds), get_std(angles), get_std(dihedrals)
        # update_topology(topology[resname], av_bonds, av_angles, av_dihedrals, std_bonds, std_angles, std_dihedrals)
        # itpio.write_itp(f"cgtools/itp/nucbonded/{name}.itp", topology[resname])   
    # save_data(all_aa_bonds, all_aa_angles, all_aa_dihedrals, all_cg_bonds, all_cg_angles, all_cg_dihedrals)
    # make_fig_one(all_aa_bonds, all_aa_angles, all_aa_dihedrals, all_cg_bonds, all_cg_angles, all_cg_dihedrals)
    

def make_figs():
    make_fig_bonded()
    


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


def persistence_length(structure):
    pass

    
def main():
    structure = make_structure_pdb(f"cif/md.pdb")
    for model in structure:
        residues = get_residues(model)
        atoms, resids = get_atoms_by_name(residues, atom_name='BB3')
        coords = get_coords(atoms)
            

if __name__ == "__main__":
    bonded_parameters()
    # make_figs()

    
    

  
