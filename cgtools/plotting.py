import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import pearsonr
from dci_dfi import percentile


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
        err = data[2]
        ax.plot(x, y, label=label, color=color)
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
        ax.set_xlabel("Residue Number", fontsize=14)
        ax.set_ylabel(self.ylabel, fontsize=14)
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
    
    def __init__(self, *args, **kwargs):
    
        super().__init__(*args, **kwargs)
        self.size       = kwargs.pop('size', (5, 5))
        self.xlabel     = kwargs.pop('xlabel', None)
        self.ylabel     = kwargs.pop('ylabel', None)
        self.cbarlabel  = kwargs.pop('cbarlabel', 'DFI')
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
        self.do_asym    = kwargs.pop('do_asym', False)
        
    def mkmask(self):
        if self.resx and self.resy:
            resxoff = [x - self.offset for x in self.resx]
            resyoff = [y - self.offset for y in self.resy]
            mask = np.ix_(resyoff, resxoff)
        else:
            mask = None
        return mask  
        
    @staticmethod    
    def apply_mask(array, mask):
        if mask:
            array = array[mask]
        return array
        
    def asym(self, mean, sem):
        if self.do_asym:
            mean = -mean + mean.T
            sem = np.sqrt(sem**2 + sem.T**2)
        return mean, sem
    
    def make_ax(self, ax, data):
        system, idx = get_system_idx(data)
        datadir = f'systems/{system}/data_{idx}'
        data_mean = os.path.join(datadir, f'{self.metric}_mean.csv')
        data_sem = os.path.join(datadir, f'{self.metric}_sem.csv')
        mean = read_data(data_mean)
        sem = read_data(data_sem)
        mean, sem = self.asym(mean, sem)
        mask = self.mkmask()
        mean = self.apply_mask(mean, mask)
        sem = self.apply_mask(sem, mask)
        sem_av = np.average(sem)
        heatmap = ax.imshow(mean, cmap=self.cmap, interpolation='nearest', vmin=self.vmin, vmax=self.vmax)
        self.set_ax_parameters(ax, data, sem_av)
        
    def set_ticks(self, ax):
        resx = self.resx
        resy = self.resy
        if resx:
            ax.set_xticks(range(0, len(resx)), labels=resx, rotation=90)
            # ax.set_xticklabels(resx)
        if resy:
            ax.set_yticks(range(0, len(resy)))
            ax.set_yticklabels(resy)   
        
    def set_ax_parameters(self, ax, data, sem_av):
        self.set_ticks(ax)
        ax.set_title(f'{data} {sem_av:.2f}', fontsize=12)
        
    def set_fig_parameters(self, fig, axs): 
        norm = Normalize(vmin=self.vmin, vmax=self.vmax)
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=self.cmap), ax=axs[-1], orientation='vertical', shrink=self.shrink, pad=0.04)
        cbar.set_ticks([self.vmin, 0.5*(self.vmin+self.vmax), self.vmax])
        cbar.set_label(self.cbarlabel, fontsize=12)
        fig.suptitle(self.title, fontsize=14)
        plt.tight_layout()
  
        
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
        
        
def plot_xvgs():
    # importing
    from plotting import Plot2D
    # plot data from each run in one figure  
    def mkfig(files, fname, **kwargs):
        datas = []
        labels = []
        for file in files:
            data = read_data(file)
            datas.append(data)
            label = file.split('/')[-3]
            labels.append(label)
        figname = os.path.join(system.pngdir, fname.replace('csv', 'png'))
        plot = Plot2D([datas], [labels], figname=figname, legend=True, loc='upper right', **kwargs)
        plot.make_plot()
    # make figures by filenames   
    def make_figs(system, fnames, **kwargs):
        for fname in fnames:
            files = system.pull_runs_files('rms_analysis', fname)
            mkfig(files, fname, **kwargs)
    # plot      
    system = CGSystem(sysdir, sysname)  
    system.runs_mean_sem('rms_analysis', 'rdf_MG.xvg')
    rdf_names = [f for f in os.listdir(system.datdir) if f.startswith('rdf')]
    make_figs(system, rdf_names)
    # rmsf_names = [f for f in os.listdir(system.initmd('mdrun_1').rmsdir) if f.startswith('rmsf')]
    # make_figs(rmsf_names)   
        
        
if __name__ == '__main__':
    pass
    
   
    

