import numpy as np
import sys
import reforge.forge.forcefields as ffs
import reforge.forge.cgmap as cgmap
from reforge.forge.topology import Topology, BondList
from reforge.forge.geometry import get_cg_bonds, get_aa_bonds
from reforge.plotting import init_figure, make_hist, plot_figure


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


def prep_data(aabonds, cgbonds, resname):
    bins = 50
    aa_dict = aabonds.categorize()
    cg_dict = cgbonds.categorize()
    if resname == 'all':
        keys = [comm.split()[1] for comm in aabonds.comms]
        keys = sorted(set(keys))
        aadatas, cgdatas = [], []
        for key in keys:
            filtered = BondList([bond for bond in aabonds if key in bond[2]])
            aadatas.append(filtered.measures)
            filtered = BondList([bond for bond in cgbonds if key in bond[2]])
            cgdatas.append(filtered.measures)
            axtitles = keys
    else:
        res_keys = [key for key in cg_dict.keys() if key.startswith(resname)]
        aadatas = [aa_dict[key].measures for key in res_keys]
        cgdatas = [cg_dict[key].measures for key in res_keys]
        axtitles = [key.split()[1] for key in res_keys]
    return aadatas, cgdatas, axtitles


def make_histograms(aabonds, cgbonds, aaparams, cgparams, resid='all', resname='all', figname='all', grid=(3, 4), figpath=f'png/test.png'):
    print(f'Plotting {resname}...', file=sys.stderr)
    # prep data for plotting 
    aadatas, cgdatas, axtitles = prep_data(aabonds, cgbonds, resid)
    # plotting 
    fig, axes = init_figure(grid=grid, axsize=(4, 4))
    for ax, aadata, cgdata, axtitle in zip(axes, aadatas, cgdatas, axtitles):
        make_hist(ax, [aadata, cgdata], [aaparams, cgparams], title=axtitle)
    plot_figure(fig, axes, figname=figname, figpath=figpath)
    print(f'Done!', file=sys.stderr)


if __name__ == "__main__":
    refpdb = 'dsRNA.pdb'
    cgpdb = 'models.pdb'
    aapdb = 'dsRNA.pdb'
    ff = ffs.martini30rna()
    reference_topology = get_reference_topology(refpdb)
    cgbonds, cgangles, cgdihs = get_cg_bonds(cgpdb, reference_topology)
    aabonds, aaangles, aadihs = get_aa_bonds(aapdb, ff, reference_topology)
    # Plotting all
    bins = 50
    aaparams = {'bins': bins, 'density': True, 'fill': True}
    cgparams = {'bins': bins, 'density': True, 'fill': False}
    make_histograms(aadihs, cgdihs, aaparams, cgparams, figname=f'Dihedrals', grid=(4, 4), figpath=f'png/dihs_all.png')
    make_histograms(aaangles, cgangles, aaparams, cgparams, figname=f'Angles', grid=(3, 4), figpath=f'png/angles_all.png')
    make_histograms(aabonds, cgbonds, aaparams, cgparams, figname=f'Distances', grid=(4, 4), figpath=f'png/bonds_all.png')
    exit()
    # Plotting by residue
    resnames = {'A': 'Adenine', 'C': 'Cytosine', 'G': 'Guanine', 'U': 'Uracil'}
    for resid, resname in resnames.items():
        make_histograms(aadihs, cgdihs, aaparams, cgparams, resid, resname, figname=f'Dihedrals {resname}', 
            grid=(3, 4), figpath=f'png/dihs_{resid}.png')
        make_histograms(aaangles, cgangles, aaparams, cgparams, resid, resname, figname=f'Angles {resname}', 
            grid=(2, 4), figpath=f'png/angles_{resid}.png')
        make_histograms(aabonds, cgbonds, aaparams, cgparams, resid, resname, figname=f'Distances {resname}', 
            grid=(3, 4), figpath=f'png/bonds_{resid}.png')

    

    






        
   
