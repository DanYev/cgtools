import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from reforge import io

##########################
## Plotting ##
##########################


def init_figure(grid=(2, 3), axsize=(4, 4), **kwargs):
    """Instantiate a figure.

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
    params - list of kwargs dictionary for the ax
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
    params - list of kwargs dictionary for the ax
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
    params - list of kwargs dictionary for the ax
    """
    if not params:
        params = [{} for x in xs]
    for x, y, err, param in zip(xs, ys, errs, params):
        ax.plot(x, y, **param)
        ax.fill_between(x, y - err, y + err, **kwargs)


def make_heatmap(ax, data, **params):
    ax.imshow(data, **params)


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
    # Add vertical lines
    for line_pos, label_pos, label in zip(line_positions, label_positions, labels):
        ax.axvline(x=line_pos, color='k', linestyle=':', label=None)
        ax.text(label_pos/max_line_pos-0.008, 1.03, label, transform=ax.transAxes, 
            rotation=90, fontsize=14) 
    for spine in ax.spines.values(): # Increase spine width for a bolder look
        spine.set_linewidth(1.5)
    legend = ax.legend(fontsize=14, frameon=False) # Add a legend with custom font size and no frame
   # Autoscale the view to the data
    ax.relim()         # Recalculate limits based on current artists
    ax.autoscale_view()  # Update the view to the recalculated limits
    ax.margins(0) # Remove padding around the data


def set_hm_parameters(ax, xlabel=None, ylabel=None, axtitle=None):
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
    # Add grid
    for line_pos, label_pos, label in zip(line_positions, label_positions, labels):
        ax.axvline(x=line_pos, color='k', linestyle=':', label=None)
        ax.axhline(y=line_pos, color='k', linestyle=':', label=None)
        ax.text(label_pos/max_line_pos-0.008, 1.01, label, transform=ax.transAxes, 
            rotation=90, fontsize=14) 
        ax.text(1.01, 0.992-label_pos/max_line_pos, label, transform=ax.transAxes, 
            fontsize=14) 
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    legend = ax.legend(fontsize=14, frameon=False)
   # Autoscale the view to the data
    ax.relim()         # Recalculate limits based on current artists
    ax.autoscale_view()  # Update the view to the recalculated limits
    ax.margins(0) # Remove padding around the data


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
    ax.set_xlabel('Time (ns)', fontsize=16)
    ax.set_ylabel('RMSD (Angstrom)', fontsize=16)
    plot_figure(fig, ax, figname=mdsys.sysname.upper() , figpath=f'{mdsys.pngdir}/rmsd.png',)


def plot_dci(mdsys):
    logger.info("Plotting DCI")
    # Pulling data
    datas, errs = pull_data('dci*')
    param = {'lw':2}
    datas = [data for data in datas]
    data = datas[0]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 12))
    make_heatmap(ax, data, cmap='bwr', interpolation=None, vmin=0, vmax=2)
    set_hm_parameters(ax, xlabel='Residue', ylabel='Residue')
    plot_figure(fig, ax, figname=f"{mdsys.sysname.upper()} DCI", figpath=f'{mdsys.pngdir}/dci.png',)


def plot_asym(mdsys):
    logger.info("Plotting DCI asym")
    # Pulling data
    datas, errs = pull_data('asym*')
    param = {'lw':2}
    datas = [data for data in datas]
    data = datas[0]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 12))
    make_heatmap(ax, data, cmap='bwr', interpolation=None, vmin=-1, vmax=1)
    set_hm_parameters(ax, xlabel='Residue', ylabel='Residue')
    plot_figure(fig, ax, figname=f"{mdsys.sysname.upper()} DCI-ASYM", figpath=f'{mdsys.pngdir}/asym.png',)


def plot_pert(mdsys):
    logger.info("Plotting pertmat")
    # Pulling data
    datas, errs = pull_data('pert*')
    param = {'lw':2}
    datas = [data for data in datas]
    data = datas[0]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 12))
    make_heatmap(ax, data, cmap='bwr', interpolation=None, vmin=0, vmax=2)
    set_hm_parameters(ax, xlabel='Residue', ylabel='Residue')
    plot_figure(fig, ax, figname=f"{mdsys.sysname.upper()} PertMat", figpath=f'{mdsys.pngdir}/pert.png',)


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


def plot_segment_asym(mdsys, segid):
    logger.info("Plotting %s ASYM", segid)
    # Pulling data
    datas, errs = pull_data(f'gasym_{segid}*')
    param = {'lw':2}
    xs = [np.arange(len(data)) for data in datas]
    datas = [data for data in datas]
    errs = [err for err in errs]
    params = [param for data in datas] 
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_errorbar(ax, xs, datas, errs, params, alpha=0.7)
    set_ax_parameters(ax, xlabel='Residue', ylabel='DCI-ASYM')
    plot_figure(fig, ax, figname=mdsys.sysname.upper() + " " + segid.upper(), 
        figpath=f'{mdsys.pngdir}/gasym_{segid}.png',)


##########################
## Animations ##
##########################


def response_1(mat, norm=1):
    resp = np.average(mat, axis=1)
    resp = np.sqrt(resp**2)
    resp = resp.reshape((len(resp) // 3, 3))
    resp = np.sum(resp, axis=1)
    resp /= norm
    return resp


def response_2(mat, norm=1):
    resp = np.average(mat**2, axis=1)
    resp = np.sqrt(resp)
    resp = resp.reshape((len(resp) // 3, 3))
    resp = np.sum(resp, axis=1)
    resp /= norm
    return resp


def response_2_2d(mat, norm=1):
    resp = mat**2
    resp = resp.reshape((resp.shape[0] // 3, resp.shape[1] // 3, 3, 3))
    resp = np.sum(resp, axis=(2, 3))
    resp = np.sqrt(resp)
    resp /= norm
    return resp


def response_force(mat_t):
    nx = mat_t.shape[0]
    ny = mat_t.shape[1]
    nt = mat_t.shape[2]
    t = np.arange(nt)
    k = 0.01
    t = np.sin(2 * np.pi * k * t)
    dt = 2 * np.pi * k * np.cos(2 * np.pi * k * t)
    f = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    f = np.tile(f, (nx // 3, ny // 3))
    force = np.einsum("ij,k->ijk", f, t)
    dforce = np.einsum("ij,k->ijk", f, dt)
    conv = lrt.gfft_conv(dforce, mat_t)
    mat_0 = mat_t[:, :, 0][:, :, None]
    force_0 = force[:, :, 0][:, :, None]
    resp = mat_0 * force - mat_t * force_0 + conv
    # resp  = mat_0 - mat_t
    return resp


def response(mat, norm=1):
    return response_2_2d(mat, norm)


def make_1d_data(infile, nframes=1000):
    print(f"Processing {infile}", file=sys.stderr)
    arrays = []
    mat_t = np.load(infile)
    mat_t = np.swapaxes(mat_t, 0, 1)
    # mat_t = response(mat_t)
    # mat_t -= np.average(mat_t, axis=-1, keepdims=True)
    mat_0 = mat_t[:, :, 0]
    resp_0 = response(mat_0, 1)
    norm = np.average(resp_0)
    if nframes > mat_t.shape[2]:  # Plot only the valid part
        nframes = mat_t.shape[2]
    # arrays.append(np.zeros(pertmat_0.shape[0]))
    for i in range(0, nframes):
        mat = mat_t[:, :, i]
        resp = response(mat, 1)
        arrays.append(resp)
    # arrays /= norm
    print("Finished computing arrays", file=sys.stderr)
    return arrays


def make_2d_data(infile, nframes=1000):
    print(f"Processing {infile}", file=sys.stderr)
    matrices = []
    mat_t = np.load(infile)
    mat_t = np.swapaxes(mat_t, 0, 1)
    mat_0 = mat_t[:, :, 0]
    resp_0 = response(mat_0, 1)
    norm = np.average(resp_0)
    if nframes > mat_t.shape[2]:  # Plot only the valid part
        nframes = mat_t.shape[2]
    for i in range(0, nframes):
        mat = mat_t[:, :, i]
        resp = response(mat, 1)
        matrices.append(resp)
    print("Finished computing matrices", file=sys.stderr)
    return matrices


def make_heatmap_td(data, outfile="png/heatmap.png"):
    print("Making a heatmap", file=sys.stderr)
    fig, ax = plt.subplots()
    img = ax.imshow(
        data,
        cmap="bwr",
        vmin=0.0,
        vmax=2.0,
    )
    xmax = data.shape[0] - 1
    ymax = data.shape[1] - 1
    line_positions = [9, 14, 761, 839, 982, 985, 1332, 1334, 1367, 1380]
    # ax.vlines(line_positions, ymin=0, ymax=xmax, colors='black', linestyles='dashed', linewidth=0.8, alpha=1.0)
    ax.hlines(
        line_positions,
        xmin=0,
        xmax=ymax,
        colors="black",
        linestyles="dashed",
        linewidth=0.8,
        alpha=1.0,
    )
    fig.savefig(outfile)
    return fig, img


def make_plot_td(datas, labels, outfile="png/test.png"):
    print("Making a plot", file=sys.stderr)
    fig, ax = plt.subplots(figsize=(12, 5))
    lines = []
    for data, label in zip(datas, labels):
        x = np.arange(1, len(data[0]) + 1)  # Create x-axis values from 0 to len(data)-1
        (line,) = ax.plot(x, data[0], label=label)  # Initial plot
        lines.append(line)
    ymin, ymax = 0, 3
    xmin, xmax = 1, 1500
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    # ax.set_ylim(np.min(data), np.max(data))  # Set y-axis limits dynamically
    line_positions = [10, 15, 762, 840, 983, 986, 1333, 1335, 1368, 1381]
    line_labels = ["D10", "S15", "E762", "H840", "H983", "D986", "R1333", "R1335"]
    ax.vlines(
        line_positions,
        ymin=ymin,
        ymax=ymax,
        colors="black",
        linestyles="dashed",
        linewidth=0.8,
        alpha=1.0,
    )
    # ax.hlines(line_positions, xmin=xmin, xmax=xmax, colors='black', linestyles='dashed', linewidth=0.8, alpha=1.0)
    ax.set_xlabel("Residue")
    ax.set_ylabel("Response")
    ax.legend(loc="upper left")
    plt.tight_layout()
    # ax.set_title("Constant Force Perturbation")
    fig.savefig(outfile)
    return fig, ax, lines


def make_plot_t_td(datas, labels, outfile="png/test.png", dt=0.2):
    print("Making a plot", file=sys.stderr)
    # residues = [9, 14, 761, 839, 982, 985, ]
    resp_residues = list(range(6))
    pert_residues = [1332, 1334]
    resls = [
        "D10",
        "S15",
        "E762",
        "H840",
        "H983",
        "D986",
    ]  #  ['R1333', 'R1335']
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["k", "red"]
    for data, label, color in zip(datas, labels, colors):
        data = np.array(data)
        for residue, resl in zip(residues, resls):
            x = (
                np.arange(data.shape[0]) * dt
            )  # Create x-axis values from 0 to len(data)-1
            ax.plot(
                x[:400], data[:400, residue], label=f"{label}_{resl}", lw=2, color=color
            )  # Initial plot
    ymin, ymax = 0, 1.2
    ax.set_ylim(ymin, ymax)
    # xmin, xmax = 1, 1368
    # ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Response")
    ax.legend(loc="upper right")
    plt.tight_layout()
    # ax.set_title("Constant Force Perturbation")
    fig.savefig(outfile)


def make_plot_t_2d(datas, labels, outfile="png/test.png", dt=0.2, nframes=750):
    print("Making a plot", file=sys.stderr)
    # residues = [9, 14, 761, 839, 982, 985, ]
    resp_residues = list(range(6))
    pert_residues = [1332, 1334]
    resls = [
        "D10",
        "S15",
        "E762",
        "H840",
        "H983",
        "D986",
    ]  #  ['R1333', 'R1335']
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["k", "red", "grey", "blue", "yellow"]
    lss = ["solid", "dashed"]
    for data, label, ls in zip(datas, labels, lss):
        data = np.array(data)
        for i, color, resl in zip(range(6), colors, resls):
            x = (
                np.arange(data.shape[0]) * dt
            )  # Create x-axis values from 0 to len(data)-1
            ax.plot(
                x[:nframes],
                data[:nframes, 1333, i],
                label=f"{label}_{resl}",
                lw=2,
                color=color,
                linestyle=ls,
            )  # Initial plot
    ymin, ymax = 0, 3
    ax.set_ylim(ymin, ymax)
    # xmin, xmax = 1, 1368
    # ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Response")
    ax.legend(loc="upper right")
    plt.tight_layout()
    # ax.set_title("Constant Force Perturbation")
    fig.savefig(outfile)


def animate_1d(fig, ax, lines, datas, outfile="data/ani1d.mp4", dt=0.2):
    print("Working on animation", file=sys.stderr)

    def update(frame):
        for line, data in zip(lines, datas):
            line.set_ydata(data[frame])  # Update y-values for each frame
            ax.set_title(f"Time {dt * frame:.2f}, ns")
        return tuple(lines)

    ani = animation.FuncAnimation(
        fig, update, frames=len(datas[0]), interval=50, blit=False
    )
    ani.save(outfile, writer="ffmpeg")  # Save as mp4
    print("Done!", file=sys.stderr)


def animate_2d(fig, img, data, outfile="data/ani2d.mp4", dt=0.2):
    print("Working on animation", file=sys.stderr)

    def update(frame):
        img.set_array(data[frame])
        ax.set_title(f"Time {dt * frame:.2f}, ps")
        return img

    ani = animation.FuncAnimation(
        fig, update, frames=len(data), interval=50, blit=False
    )
    ani.save(outfile, writer="ffmpeg")  # Save as mp4
    print("Done!", file=sys.stderr)


def make_animation(infile, mode="1d", tag="pv", outfile=None):
    print("Working on movies", file=sys.stderr)
    if not outfile:
        outfile = f"data/{mode}_{tag}_{sysname}_{runname}.mp4"
    if mode == "1d":
        data = make_1d_data(infile)
        fig, img = make_plot(data[0])
    if mode == "2d":
        data = make_2d_data(infile)
        fig, img = make_heatmap(data[0])
    animate(fig, img, data, mode=mode, outfile=outfile)


def mkfigure(datas, labels):
    # figure = Heatmap(datas, labels, legend=True, loc='upper right', vmin=1, vmax=5, cmap='bwr',
    #                 shape=(2, 1), size=(6.2, 5), fontsize=13,
    #                 makecbar=False, cbarlabel=None,
    #                 xlabel='Perturbed Residue', ylabel='Coupled Residue')
    figure = Graph(
        datas,
        labels,
        legend=True,
        loc="upper right",
        size=(6.2, 5),
        fontsize=13,
        xlabel="Perturbed Residue",
        ylabel="Coupled Residue",
    )
    return figure


def make_1d_plots(sysdir, sysnames):
    print("Plotting", file=sys.stderr)
    datas = []
    for n, sysname in enumerate(sysnames):
        system = gmxSystem(sysdir, sysname)
        infile = os.path.join(system.datdir, f"corr_pp_slow.npy")
        data = make_2d_data(infile, nframes=2000)
        np.save(f"data/arr_{n}.npy", data)
        datas.append(data)
    # datas = [np.load('data/arr_0.npy'), np.load('data/arr_1.npy'),]
    averages = [np.average(data[0]) for data in datas]
    av = np.average(averages)
    datas = [data / av for data in datas]
    outfile = os.path.join("png", f'pp_{"_".join(sysnames)}.mp4')
    fig, ax, lines = make_plot_t_2d(datas, sysnames, outfile="png/test.png")
    animate_1d(fig, ax, lines, datas, outfile, dt=0.2)


# def make_2d_plots(sysdir, sysnames):
#     print("Plotting", file=sys.stderr)
#     datas = []
#     # datas = [[np.array([1, 2, 3]), np.array([1, 2.5, 3])], [np.array([3, 4, 5]), np.array([1, 2.5, 3])]]
#     for n, sysname in enumerate(sysnames):
#         system = gmxSystem(sysdir, sysname)
#         infile = os.path.join(system.datdir, f'corr_pp_slow.npy')
#         data = make_2d_data(infile, nframes=5000)
#         datas.append(data)
#     outfile = os.path.join('png', f'fast_{"_".join(sysnames)}.mp4')
#     fig, ax, lines = make_plot(datas, sysnames, outfile="png/test.png")
#     animate_1d(fig, ax, lines, datas, outfile, dt=0.2)
