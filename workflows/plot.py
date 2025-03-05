import os
import numpy as np
import pandas as pd
import sys
from reforge import io
from reforge.mdsystem.mdsystem import MDSystem
from reforge.mdm import percentile
from reforge.plotting import *
from reforge.utils import logger


def pull_data(metric):
    files = io.pull_files(mdsys.datdir, metric)
    datas = [np.load(file) for file in files if '_av' in file]
    errs = [np.load(file) for file in files if '_err' in file]
    return datas, errs


def set_ax_parameters(ax, xlabel=None, ylabel=None, axtitle=None):
    """
    ax - matplotlib ax object
    """
    # Set axis labels and title with larger font sizes
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(axtitle, fontsize=16)
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5, width=1.5)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False)
    line_positions, label_positions, labels = grid_labels(mdsys)
    max_line_pos = max(line_positions)
    # ax.set_xticks(label_positions)
    # ax.set_xticklabels(labels, rotation=90)
    # Add vertical lines
    for line_pos, label_pos, label in zip(line_positions, label_positions, labels):
        ax.axvline(x=line_pos, color='k', linestyle=':', label=None)
        ax.text(label_pos/max_line_pos-0.008, 1.03, label, transform=ax.transAxes, 
            rotation=90, fontsize=14) 
    # Increase spine width for a bolder look
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # Add a legend with custom font size and no frame
    legend = ax.legend(fontsize=14, frameon=False)
   # Autoscale the view to the data
    ax.relim()         # Recalculate limits based on current artists
    ax.autoscale_view()  # Update the view to the recalculated limits
    # Remove padding around the data
    ax.margins(0)


def grid_labels(mdsys):
    atoms = io.pdb2atomlist(mdsys.solupdb)
    backbone_anames = ["BB"]
    bb = atoms.mask(backbone_anames, mode='name')
    bb.renum() # Renumber atids form 0, needed to mask numpy arrays
    groups = bb.segments.atids # mask for the arrays
    labels = [segids[0] for segids in bb.segments.segids]
    line_positions = [group[0] for group in groups]
    line_positions.append(groups[-1][-1])
    label_positions = [group[len(group)//2] for group in groups]
    return line_positions, label_positions, labels


def plot_dfi(mdsys):
    logger.info("Plotting DFI")
    # Pulling data
    datas, errs = pull_data('dfi*')
    param = {'lw':2}
    xs = [np.arange(len(data)) for data in datas]
    datas = [data for data in datas]
    errs = [err for err in errs]
    params = [param for data in datas]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_errorbar(ax, xs, datas, errs, params, alpha=0.7)
    set_ax_parameters(ax, xlabel='Residue', ylabel='DFI')
    plot_figure(fig, ax, figname=mdsys.sysname.upper(), figpath=f'{mdsys.pngdir}/dfi.png',)


def plot_pdfi(mdsys):
    logger.info("Plotting %DFI")
    # Pulling data
    datas, errs = pull_data('dfi*')
    param = {'lw':2}
    xs = [np.arange(len(data)) for data in datas]
    datas = [percentile(data) for data in datas]
    params = [param for data in datas]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, datas, params)
    set_ax_parameters(ax, xlabel='Residue', ylabel='%DFI')
    plot_figure(fig, ax, figname=mdsys.sysname.upper(), figpath=f'{mdsys.pngdir}/pdfi.png',)


def plot_rmsf(mdsys):
    logger.info("Plotting RMSF")
    # Pulling data
    datas, errs = pull_data('rmsf*')
    xs = [np.arange(len(data)) for data in datas]
    datas = [data*10 for data in datas]
    errs = [err*10 for err in errs]
    params = [{'lw':2} for data in datas]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_errorbar(ax, xs, datas, errs, params, alpha=0.7)
    set_ax_parameters(ax, xlabel='Residue', ylabel='RMSF (Angstrom)')
    plot_figure(fig, ax, figname=mdsys.sysname.upper(), figpath=f'{mdsys.pngdir}/rmsf.png',)


def plot_rmsd(mdsys):
    logger.info("Plotting RMSD")
    # Pulling data
    files = io.pull_files(mdsys.mddir, 'rmsd*npy')
    datas = [np.load(file) for file in files]
    labels = [file.split('/')[-3] for file in files]
    xs = [data[0]*1e-3 for data in datas]
    ys = [data[1]*10 for data in datas]
    params = [{'lw':2, 'label':label} for label in labels]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, ys, params)
    set_ax_parameters(ax, xlabel='Time (ns)', ylabel='RMSD (Angstrom)')
    plot_figure(fig, ax, figname=mdsys.sysname.upper() , figpath=f'{mdsys.pngdir}/rmsd.png',)


def plot_dci(mdsys):
    logger.info("Plotting DCI")
    # Pulling data
    datas, errs = pull_data('dci*')
    param = {'lw':2}
    datas = [data for data in datas]
    data = datas[0]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(6, 6))
    make_heatmap(ax, data, cmap='bwr', interpolation=None, vmin=0, vmax=2)
    set_ax_parameters(ax, xlabel='Residue', ylabel='Residue')
    plot_figure(fig, ax, figname=f"{mdsys.sysname.upper()} DCI", figpath=f'{mdsys.pngdir}/dci.png',)


def plot_asym(mdsys):
    logger.info("Plotting DCI asym")
    # Pulling data
    datas, errs = pull_data('asym*')
    param = {'lw':2}
    datas = [data for data in datas]
    data = datas[0]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(6, 6))
    make_heatmap(ax, data, cmap='bwr', interpolation=None, vmin=-1, vmax=1)
    set_ax_parameters(ax, xlabel='Residue', ylabel='Residue')
    plot_figure(fig, ax, figname=f"{mdsys.sysname.upper()} DCI-ASYM", figpath=f'{mdsys.pngdir}/asym.png',)


def plot_segment_dci(mdsys, segid):
    logger.info("Plotting %s DCI", segid)
    # Pulling data
    datas, errs = pull_data(f'gdci_{segid}*')
    param = {'lw':2}
    xs = [np.arange(len(data)) for data in datas]
    datas = [data for data in datas]
    errs = [err for err in errs]
    params = [param for data in datas] 
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_errorbar(ax, xs, datas, errs, params, alpha=0.7)
    set_ax_parameters(ax, xlabel='Residue', ylabel='DCI')
    plot_figure(fig, ax, figname=mdsys.sysname.upper() + " " + segid.upper(), 
        figpath=f'{mdsys.pngdir}/gdci_{segid}.png',)


def plot_all_segments(mdsys):
    for segid in mdsys.segments:   
        plot_segment_dci(mdsys, segid)
   
    
if __name__ == '__main__':
    sysdir = 'systems' 
    sysname = 'egfr'
    mdsys = MDSystem(sysdir, sysname)
    plot_dfi(mdsys)
    # plot_pdfi(mdsys)
    # plot_dci(mdsys)
    # plot_asym(mdsys)
    # plot_rmsf(mdsys)
    # plot_rmsd(mdsys)
    # plot_all_segments(mdsys)
    # grid_labels(mdsys)

