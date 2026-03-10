import matplotlib.pyplot as plt

def plot_data(d_data):
    if d_data["component_type"] == "PSD_monitor":
        fig = display_psd(d_data)
    elif d_data["component_type"] == "TOF_monitor":
        fig = display_tof(d_data)
    elif d_data["component_type"] == "DivPos_monitor":
        fig = display_divpos(d_data)
    else:
        fig = get_figure_common(d_data)
    return fig

def get_figure_common(d_data):
    fig = plt.figure()
    ax = fig.add_axes((0,0,1,1))
    ncount = int(d_data["ncount"].sum())
    ax.set_title(d_data["component_name"] + f" at ({d_data['position'][0]:.3f} {d_data['position'][1]:.3f} {d_data['position'][2]:.3f})m Ncount {ncount:}")
    return fig

def display_psd(d_data):
    fig = get_figure_common(d_data)
    ax = fig.axes[0]
    np_x, np_y = d_data["X_position"], d_data["Y_position"]
    pos = ax.imshow(d_data["data"].transpose(), origin="lower", extent=(np_x.min(), np_x.max(), np_y.min(), np_y.max()),vmin=0, vmax=1e7)
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel("X position (cm)")
    ax.set_ylabel("Y position (cm)")
    return fig

def display_tof(d_data):
    fig = get_figure_common(d_data)
    ax = fig.axes[0]
    np_x, np_y = d_data["time_of_flight"], d_data["data"]
    ax.plot(np_x, np_y,"ko")
    ax.plot(np_x, np_y,"k-", alpha=0.3)
    ax.set_xlabel("Time of Flight (micro seconds)")
    ax.set_ylabel("Intensity (arb. units)")
    return fig

def display_divpos(d_data):
    fig = get_figure_common(d_data)
    ax = fig.axes[0]
    np_x, np_y = d_data["pos"], d_data["divergence"]
    pos = ax.imshow(d_data["data"].transpose(), origin="lower", extent=(np_x.min(), np_x.max(), np_y.min(), np_y.max()), aspect="auto")
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Divergence (deg.)")
    return fig