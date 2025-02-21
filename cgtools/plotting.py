import matplotlib.pyplot as plt

##########################
## Plotting ##
##########################

def init_figure(grid=(2, 3), axsize=(4, 4), **kwargs):
    """
    Instantiate a figure.
    We can modify axes separately
    """
    m, n = grid
    ax_x, ax_y = axsize
    figsize = (ax_x * n, ax_y * m)
    fig, axes = plt.subplots(m, n, figsize=figsize, **kwargs)
    return fig, axes.flatten()


def make_hist(ax, datas, params, **kwargs):
    """
    ax - matplotlib ax object
    datas - list of datas to histogram
    params - list of kwargs dictionary for the histogram
    """
    title = kwargs.pop('title', 'Ax')
    for data, param in zip(datas, params):
        ax.hist(data, **param)
    ax.set_title(title)


def plot_figure(fig, axes, figname='Title', figpath='test.png', **kwargs):
    """
    Finish plotting
    """
    fig.suptitle(figname)
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()