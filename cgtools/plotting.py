import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import pearsonr
from dci_dfi import percentile


def read_data(fpath):
    """ 
    Reads a .csv or .npy data 
    
    Input 
    ------
    fname: string
        Name of the data to read
    Output 
    ------
    data: ndarray
        Numpy array
    """
    ftype = fpath.split('.')[-1]
    if ftype == 'npy':
        try:
            data = np.load(fpath)
        except:
            raise ValueError()
    if ftype == 'csv' or ftype == 'dat':
        try:
            df = pd.read_csv(fpath, sep='\\s+', header=None)
            data = df.values
            data = np.squeeze(df.values)
        except:
            raise ValueError()
    if ftype == 'xvg':
        try:
            df = pd.read_csv(fpath, sep='\\s+', header=None)
            x = df[0].values
            y = df[1].values
            if len(df.columns) > 2:
                err = df[2].values
            else:
                err = None
        except:
            raise ValueError()
    return (x , y, err)


def flatten_list(nested_list):
    flat_list = []
    for element in nested_list:
        if isinstance(element, list):
            flat_list.extend(flatten_list(element))  # Recursively flatten sublists
        else:
            flat_list.append(element)  # Append non-list elements directly
    return flat_list
 
def plot_aa_dfi(data=f'data/extant_300_dfi_100ns_pct_chain_analysis.csv'):
    df = pd.read_csv(data)
    x = np.array(df[df.iloc[:, 1].notna()].iloc[:, 1])
    y = np.array(df[df.iloc[:, 2].notna()].iloc[:, 2])
    y = percentile(y)
    N = len(y)
    x = np.arange(672, N+672)
    nsys = 1
    fig = plt.figure(figsize=(12,4))
    plt.plot(x, y, label='AA')
    y = np.load('data/hM8_TMD_04_WT_300_dfi.npy')[:-1] # hM8_TMD_04 hTRPM8[672:-92]
    y = percentile(y)
    plt.plot(x, y, label='CG')
    plt.grid()
    plt.legend(loc='upper right')
    plt.xlabel('ResN')
    plt.ylabel('pDFI')
    plt.autoscale(tight=True)
    plt.tight_layout()
    fig.savefig(f'png/test.png')
    plt.close()
    
    
def get_mutations(data):
    datas = flatten_list(data)
    mutations = [x.split('_')[1:-2] for x in datas]
    mutations = set(flatten_list(mutations))
    if 'WT' in mutations:
        mutations.remove('WT')
    if mutations:
        positions = [int(x[1:-1]) for x in mutations]
    else:
        positions = None
    return positions, mutations


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

    def make_plot(self):
        """
        Plot the metric for each data across a grid of subplots.
        """
        ###
        fig, axs = self.make_grid()
        for ax, data, label in zip(axs, self.datas, self.labels):
            self.make_ax(ax, data, label)
        self.set_fig_parameters(fig, axs)
        fig.savefig(self.figname)
        plt.close()
        
        
class Plot2D(Figure):   
    
    def __init__(self, datas, labels, **kwargs):
        super().__init__(datas, **kwargs)
        self.labels     = labels
        self.size       = kwargs.pop('size', (12, 5))
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
        if err:
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
        ax.set_ylabel(self.ylabel, fontsize=14)
        ax.set_ylim(self.ylim)
        ax.set_xlabel("Residue Number", fontsize=14)
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
        
        
if __name__ == '__main__':
    pass
    
   
    

