import numpy as np
import sys
import cgtools.forge.forcefields as ffs
import cgtools.forge.cgmap as cgmap
from cgtools.forge.topology import Topology, BondList
from cgtools.forge.geometry import get_cg_bonds, get_aa_bonds
from cgtools.plotting import init_figure, make_hist, plot_figure


def get_reference_topology(inpdb):
    # Need to get the topology from the reference system
    print(f'Calculating the reference topology from {inpdb}...', file=sys.stderr)
    system = cgmap.read_pdb(inpdb) 
    cgmap.move_o3(system) # Need to move all O3's to the next residue. Annoying but wcyd
    topologies = []
    for chain in system.chains():
        top = Topology(ff)
        top.from_chain(chain)
        topologies.append(top)
    top = Topology.merge_topologies(topologies)
    print('Done!', file=sys.stderr)
    return top


def prep_data(bonds, resname):
    bins = 50
    params1 = {'density': False, 'fill': True}
    params2 = {'density': False, 'fill': False}
    b_dict = bonds.categorize()
    res_keys = [key for key in b_dict.keys() if key.startswith(resname)]
    datas1 = [b_dict[key].measures for key in res_keys]
    datas2 = datas1
    titles = [key.split()[1] for key in res_keys]
    return datas1, datas2, params1, params2, titles


if __name__ == "__main__":
    refpdb = 'dsRNA.pdb'
    inpdb = 'models.pdb'
    ff = ffs.martini30rna()
    reference_topology = get_reference_topology(refpdb)
    bonds, angles, dihs = get_cg_bonds(inpdb, reference_topology)
    print(f'Plotting...', file=sys.stderr)
    resnames = {'A': 'Adenine', 'C': 'Cytosine', 'G': 'Guanine', 'U': 'Uracil'}
    for resid, resname in resnames.items():
        # prep data for plotting 
        datas1, datas2, params1, params2, axtitles = prep_data(bonds, resid)
        # plotting 
        fig, axes = init_figure(grid=(3, 4), axsize=(4, 4))
        for ax, data1, data2, axtitle in zip(axes, datas1, datas2, axtitles):
            make_hist(ax, [data1, data2], [params1, params2], title=axtitle)
        plot_figure(fig, axes, figname=resname, figpath=f'test_{resid}.png')
    print(f'Done!', file=sys.stderr)
    






        
   
