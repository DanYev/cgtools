import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import pearsonr
from cgtools.lrt import percentile


class Figure:
    
    def __init__(self, datas, **kwargs):
        """
        
        """
        self.datas      = datas
        self.figname    = kwargs.pop('figname', 'fig')
        self.title      = kwargs.pop('title', None)
        self.ylabel     = kwargs.pop('ylabel', None)
        self.shape      = kwargs.pop('shape', None)

    def make_grid(self):
        """
        Create a grid layout for the number of systems in 'datas'.
        """
        nsys = len(self.datas)
        if self.shape == None:
            shapex = 1
            shapey = nsys
        else:
            shapex, shapey = self.shape
        sizex, sizey = self.size
        fig, axs = plt.subplots(shapey, shapex, sharey=False, figsize=(sizex * shapex, sizey * shapey))
        if nsys == 1:
            axs = [axs]
        return fig, axs

    def save_figure(self, path):
        """
        Plot the metric for each data across a grid of subplots.
        """
        ###
        fig, axs = self.make_grid()
        for ax, data, label in zip(axs, self.datas, self.labels):
            self.make_ax(ax, data, label)
        self.set_fig_parameters(fig, axs)
        fig.savefig(path)
        plt.close()
        
        
class Plot2D(Figure):   
    
    def __init__(self, datas, labels, **kwargs):
        super().__init__(datas, **kwargs)
        self.labels     = labels
        self.size       = kwargs.pop('size', (12, 5))
        self.xlim       = kwargs.pop('xlim', None)
        self.ylim       = kwargs.pop('ylim', None)
        self.xscale     = kwargs.pop('xscale', 1.0)
        self.yscale     = kwargs.pop('yscale', 1.0)
        self.legend     = kwargs.pop('legend', None)
        self.legend_loc = kwargs.pop('loc', None)
        self.alpha      = kwargs.pop('alpha', 0.3)
        self.domains    = kwargs.pop('domains', None)
        # self.mutations  = get_mutations(self.datas)

    def plot_graph(self, ax, data, label, color):
        """
        Plot the graph for a given metric and system's data, on a specific axis.
        """
        x = data[0]
        y = data[1]
        ax.plot(x, y, label=label, color=color)
        if data.shape[1] == 3:
            err = data[2]
            ax.fill_between(x, y - err, y + err, color=color, alpha=self.alpha)
            
    def make_ax(self, ax, subdatas, sublabels):
        colors = plt.cm.Dark2(np.arange(0, len(subdatas))) # Set1 Set2 Dark2 Accent Pastel1
        for data, label, color in zip(subdatas, sublabels, colors):
            self.plot_graph(ax, data, label, color)
        self.set_ax_parameters(ax, subdatas)

    def set_ax_parameters(self, ax, data):
        """
        Set common plotting parameters such as labels and title.
        """
        # if legend:
        #     ax.legend(frameon=False, fontsize=12, loc=loc)
        ax.autoscale(tight=True)
        ax.grid()
        ax.set_xlabel("Residue Number", fontsize=18)
        ax.set_ylabel(self.ylabel, fontsize=18)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        # ax.set_title(data, fontsize=14)
        ax.legend(loc='best')  
        # self.annotate_domains(ax)
        
    def set_fig_parameters(self, fig, axs):
        plt.tight_layout()
        
    def annotate_domains(self, ax):
        if self.mutations[0]:
            for pos in self.mutations[0]:
                ax.axvline(pos, color='gray') # 
        if self.domains:
            for domain in self.domains:
                ax.axvspan(domain[0], domain[1], color='gray', alpha=0.3, label=domain[2])

        
        
class HeatMap(Figure):
    
    def __init__(self, datas, labels, **kwargs):
        super().__init__(datas, **kwargs)
        self.labels     = labels
        self.size       = kwargs.pop('size', (5, 5))
        self.xlabel     = kwargs.pop('xlabel', None)
        self.ylabel     = kwargs.pop('ylabel', None)
        self.makecbar   = kwargs.pop('makecbar', False)
        self.cbarlabel  = kwargs.pop('cbarlabel', None)
        self.ylim       = kwargs.pop('ylim', None)
        self.loc        = kwargs.pop('loc', None)
        self.xscale     = kwargs.pop('xscale', 1.0)
        self.yscale     = kwargs.pop('yscale', 1.0)
        self.vmin       = kwargs.pop('vmin', 0.0)
        self.vmax       = kwargs.pop('vmax', 2.0)
        self.shrink     = kwargs.pop('shrink', 1.0)
        self.resx       = kwargs.pop('resx', None)
        self.resy       = kwargs.pop('resy', None)
        self.offset     = kwargs.pop('offset', 1)
        self.cmap       = kwargs.pop('cmap', 'coolwarm')
        self.fontsize   = kwargs.pop('fontsize', 18)
        
    def make_ax(self, ax, subdatas, sublabels):
        for data, label in zip(subdatas, sublabels):
            heatmap = ax.imshow(data, cmap=self.cmap, interpolation='nearest', vmin=self.vmin, vmax=self.vmax)
        self.set_ax_parameters(ax, data)
        
    def set_ticks(self, ax):
        resx = self.resx
        resy = self.resy
        if resx:
            ax.set_xticks(range(0, len(resx)), labels=resx, rotation=90, fontsize=self.fontsize)
            ax.set_xticklabels(resx)
        if resy:
            ax.set_yticks(range(0, len(resy)), fontsize=self.fontsize)
            ax.set_yticklabels(resy)  
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(axis='both', labelsize=self.fontsize-2) 
        
    def set_ax_parameters(self, ax, data):
        self.set_ticks(ax)
        if self.xlabel:
            ax.set_xlabel(self.xlabel, fontsize=self.fontsize)
        if self.ylabel:
            ax.set_ylabel(self.ylabel, fontsize=self.fontsize)
        # ax.set_title(f'{data}', fontsize=self.fontsize+2)
        
    def set_fig_parameters(self, fig, axs): 
        norm = Normalize(vmin=self.vmin, vmax=self.vmax)
        if self.makecbar:
            cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=self.cmap), ax=axs[-1], orientation='vertical', shrink=self.shrink, pad=0.04)
            cbar.set_ticks([self.vmin, 0.5*(self.vmin+self.vmax), self.vmax])
            cbar.set_ticklabels([0, 1, 2])
            cbar.set_label(self.cbarlabel, fontsize=self.fontsize, labelpad=5)
        fig.suptitle(self.title, fontsize=self.fontsize)
        plt.tight_layout()
        # fig.set_constrained_layout(True)
  
        
################################################################################
# Some functions to make figures with CGSystem data
################################################################################    

def plot_mean_sem(files, figpath, **kwargs):
    """
    Makes a figure with .csv files in datdir containing Mean and SEM 
    """
    datas = [pd.read_csv(f, header=None) for f in files]
    labels = [f.split('/')[-3] for f in files]
    plot = Plot2D([datas], [labels], legend=True, loc='upper right', **kwargs)
    plot.save_figure(figpath)
    
    
def plot_each_mean_sem(fpaths, system, **kwargs):
    """
    Do 'plot_mean_sem' for a selection of files for a system
    """
    for fpath in fpaths:
        figname = fpath.split('/')[-1].replace('csv', 'png')
        figpath = os.path.join(system.pngdir, figname)
        plot_mean_sem([fpath], figpath, **kwargs)
        
        
def plot_xvg(files, figpath, **kwargs):
    """
    Makes a figure with .xvg files generated by GROMACS
    """  
    datas = [pd.read_csv(f, sep='\\s+', header=None, usecols=[0, 1]) for f in files]
    labels = [f.split('/')[-3] for f in files]
    plot = Plot2D([datas], [labels], legend=True, loc='upper right', **kwargs)
    plot.save_figure(figpath)    
    
    
def plot_heatmaps(files, figpath, **kwargs):
    datas = [[pd.read_csv(f, sep=',', header=None)] for f in files]
    labels = [[f.split('/')[-3]] for f in files]
    plot = HeatMap(datas, labels, **kwargs)
    plot.save_figure(figpath)   
    
    
if __name__ == '__main__':
    pass
    
   
    

