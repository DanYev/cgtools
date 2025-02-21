import numpy as np
import sys
import cgtools.forge.forcefields as ffs
import cgtools.forge.cgmap as cgmap
from cgtools.forge.topology import Topology, BondList
from cgtools.forge.geometry import get_cg_bonds, init_figure, make_hist, plot_figure


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
    print(f'Calculating the reference topology from {inpdb}...', file=sys.stderr)
    system = cgmap.read_pdb(inpdb) 
    cgmap.move_o3(system) # Need to move all O3's to the next residue. Annoying but wcyd
    topologies = []
    for chain in system.chains():
        top = process_chain(chain, ff)
        topologies.append(top)
    top = merge_topologies(topologies)
    print('Done!', file=sys.stderr)
    return top


def prep_data(bonds):
    plotparams = [ {'density': False, 'fill': True}, 
                    {'density': False, 'fill': False}]
    b_dict = bonds.categorize()
    keys = list(b_dict.keys())
    idx = 0
    key = keys[idx]
    bonds = b_dict[key]
    params = bonds.parameters
    data1 = [param[1] for param in params]
    data2 = np.array(data1) * 1.1
    datas = [data1, data2]
    return datas, plotparams


if __name__ == "__main__":
    refpdb = 'dsRNA.pdb'
    inpdb = 'models.pdb'
    ff = ffs.martini30rna()
    reference_topology = get_reference_topology(refpdb)
    bonds, angles, dihs = get_cg_bonds(inpdb, reference_topology)
    # prep data for plotting 
    datas, params = prep_data(bonds)
    alsit = [[datas, params], [datas, params], [datas, params], ]
    # plotting 
    fig, axes = init_figure(grid=(2, 2), axsize=(4, 4))
    for ax, (datas, params) in zip(axes, alsit):
        make_hist(ax, datas, params, title='ax')
    plot_figure(figpath='test.png')
    






        
   
