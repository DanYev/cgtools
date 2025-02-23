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
    return fig, axes


def make_hist(ax, datas, params=None):
    """
    ax - matplotlib ax object
    datas - list of datas to histogram
    params - list of kwargs dictionary for the histogram
    """
    if not params:
        params = [{} for data in datas]
    for data, param in zip(datas, params):
        ax.hist(data, **param)


def make_plot(ax, xs, ys, params=None):
    """
    ax - matplotlib ax object
    xs - list of x coords
    ys - list of y coords
    params - list of kwargs dictionary for the histogram
    """
    if not params:
        params = [{} for x in xs]
    for x, y, param in zip(xs, ys, params):
        ax.plot(x, y, **param)


def make_errorbar(ax, xs, ys, errs, params=None, **kwargs):
    """
    ax - matplotlib ax object
    xs - list of x coords
    ys - list of y coords
    errs - list of errors
    params - list of kwargs dictionary for the histogram
    """
    if not params:
        params = [{} for x in xs]
    for x, y, err, param in zip(xs, ys, errs, params):
        ax.plot(x, y, **param)
        ax.fill_between(x, y - err, y + err, **kwargs)


def make_heatmap(ax, data, **params):
    ax.imshow(data, **params)


def set_ax_parameters(ax, xlabel='Time (s)', ylabel='Amplitude', axtitle='Axtitle'):
    """
    ax - matplotlib ax object
    """
    # Set axis labels and title with larger font sizes
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(axtitle, fontsize=16)
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12, direction='in', length=5, width=1.5)
    # Increase spine width for a bolder look
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # Add a legend with custom font size and no frame
    legend = ax.legend(fontsize=12, frameon=False)
    # Optionally, add gridlines
    ax.grid(True, linestyle='--', alpha=0.5)

        

def plot_figure(fig, axes, figname=None, figpath='png/test.png', **kwargs):
    """
    Finish plotting
    """
    fig.suptitle(figname, fontsize=18)
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()