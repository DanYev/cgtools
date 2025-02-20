import cgtools.forge.forcefields as ffs
import cgtools.forge.cgmap as cgmap
from cgtools.forge.topology import Topology
from cgtools.forge.geometry import get_distance, get_angle, get_dihedral


def process_chain(chain, ff):
    sequence = [residue.resname for residue in chain] # So far only need sequence for the topology
    top = Topology(forcefield=ff, sequence=sequence)
    top.process_atoms() # Adds itp atom objects to the topology list 
    top.process_bb_bonds() # Adds bb bond objects to the topology list 
    top.process_sc_bonds() # Adds sc bond objects to the topology list 
    return top


def merge_topologies(topologies):
    top = topologies.pop(0)
    if topologies:
        for new_top in topologies:
            top += new_top
    return top


def get_reference_topology(inpdb):
    # Need to get the topology from the reference system
    system = cgmap.read_pdb(inpdb) 
    cgmap.move_o3(system) # Need to move all O3's to the next residue. Annoying but wcyd
    topologies = []
    for chain in system.chains():
        top = process_chain(chain, ff)
        topologies.append(top)
    top = merge_topologies(topologies)
    return top


def calc_bonds(model, bonds): # connectivities
    atoms = model.atoms()
    conns = bonds.connectivities
    comms = bonds.comments
    pairs = [(atoms[i - 1], atoms[j - 1]) for i, j in conns]
    vecs_list = [(a1.vec, a2.vec) for a1, a2 in pairs]
    dists = [get_distance(*vecs) for vecs in vecs_list]
    resnames = [a1.resname for a1, a2 in pairs]
    result = list(zip(dists, resnames, comms)) # [(dist, resname, comm) for dist, resname, comm in zip(dists, resnames, comms)]
    return result


def calc_angles(model, angles): 
    atoms = model.atoms()
    conns = angles.connectivities
    comms = angles.comments
    triplets = [(atoms[i - 1], atoms[j - 1], atoms[k - 1]) for i, j, k in conns]
    vecs_list = [(a1.vec, a2.vec, a3.vec) for a1, a2, a3 in triplets]
    angles = [get_angle(*vecs) for vecs in vecs_list]
    resnames = [a1.resname for a1, a2, a3 in triplets]
    result = list(zip(angles, resnames, comms)) # [(dist, resname, comm) for dist, resname, comm in zip(dists, resnames, comms)]
    return result


def calc_dihs(model, dihs): 
    atoms = model.atoms()
    conns = dihs.connectivities
    comms = dihs.comments
    quads = [(atoms[i - 1], atoms[j - 1], atoms[k - 1], atoms[l - 1]) for i, j, k, l in conns]
    vecs_list = [(a1.vec, a2.vec, a3.vec, a4.vec) for a1, a2, a3, a4 in quads]
    dihs = [get_dihedral(*vecs) for vecs in vecs_list]
    resnames = [a2.resname for a1, a2, a3, a4 in quads]
    result = list(zip(dihs, resnames, comms)) # [(dist, resname, comm) for dist, resname, comm in zip(dists, resnames, comms)]
    return result


if __name__ == "__main__":
    refpdb = 'dsRNA.pdb'
    inpdb = 'models.pdb'
    ff = ffs.martini30rna()
    top = get_reference_topology(refpdb)
    system = cgmap.read_pdb(inpdb)
    for model in system:
        bonds = calc_bonds(model, top.bonds)
        angles = calc_angles(model, top.angles)
        dihs = calc_dihs(model, top.dihs)
        print(dihs)




        
   
