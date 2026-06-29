from ui import *
from state import STATE, STATE_VANADIUM
import os
import plopp
import scipp as sc
import numpy
import pandas as pd
from scipy.ndimage import label, center_of_mass, sum as ndi_sum

from IPython.display import display, clear_output, HTML

import matplotlib.pyplot as plt

import magic_scipp
import read_h5
import get_ub
import magic_graphs
import peak_find
import operations_with_da
import integrate_peaks

def assign_dg_to_da_coords(dg: sc.DataGroup, da: sc.DataArray, prefix:str=""):
    for s_key in dg.keys():
        da.coords[f"{prefix}_{s_key}"] = dg[s_key]

def load_file(b):
    with output_data:
        clear_output()
        peaks_output.clear_output()
        indexation_output.clear_output()
        display(spinner)

        STATE.clear()

        fig_rbuttons.layout.display = "none"
        display_button.layout.display = "none"
        state_dropdown.layout.display = "none"
        display_da_button.layout.display = "none"
        hide_output_data_button.layout.display = "none"
        # norm_rbuttons.layout.display = "none"
        # # find_peaks_button.layout.display = "none"
        # find_peaks_hist_button.layout.display = "none"
        # cell_a.layout.display = "none"
        # cell_b.layout.display = "none"
        # cell_c.layout.display = "none"
        # cell_alpha.layout.display = "none"
        # cell_beta.layout.display = "none"
        # cell_gamma.layout.display = "none"
        # indexation_button.layout.display = "none"

        file_path = fc.selected
        if file_path is None or not os.path.isfile(file_path):
            print("No file selected")
            return
        try:
            d_out = read_h5.read_magic_from_nexus(file_path)
        except Exception as e:
            print(f"Error reading file: {e}")
            return
        STATE.magic_data = d_out
        STATE.detector_a_event = d_out.get("detector_a", None)
        STATE.detector_b_event = d_out.get("detector_b", None)
        STATE.monitor_cave = d_out.get("cave_monitor", None)

        file_path = fc_vanadium.selected
        if not file_path is None and os.path.isfile(file_path):
            try:
                d_out = read_h5.read_magic_from_nexus(file_path)

                STATE_VANADIUM.magic_data = d_out
                STATE_VANADIUM.detector_a_event = d_out.get("detector_a", None)
                STATE_VANADIUM.detector_b_event = d_out.get("detector_b", None)
                STATE_VANADIUM.monitor_cave = d_out.get("cave_monitor", None)

            except Exception as e:
                print(f"Error reading file: {e}")
        clear_output()

        display_data(b)
        fig_rbuttons.layout.display = "inline-block"
        display_button.layout.display = "inline-block"
        state_dropdown.layout.display = "inline-block"
        display_da_button.layout.display = "inline-block"
        hide_output_data_button.layout.display = "inline-block"
        # norm_rbuttons.layout.display = "inline-block"
        # # find_peaks_button.layout.display = "inline-block"
        # find_peaks_hist_button.layout.display = "inline-block"


def display_data(b):
    with output_data:
        clear_output()
        display(spinner)

        # Create plopp figure
        print(fig_rbuttons.value)
        flag_display_center = True
        if fig_rbuttons.value == "3D Laue Pattern":
            d_plot = {}
            vmax = 0.
            da_det_a = STATE.detector_a_event
            if da_det_a is not None:
                da_det_a_laue = operations_with_da.da_to_laue_hist(
                    da_det_a, factor_border=0.00
                )
                vmax = max([vmax, numpy.quantile(da_det_a_laue.data.values, 0.9)])
                d_plot["detector A"] = da_det_a_laue

            da_det_b = STATE.detector_b_event
            if da_det_b is not None:
                da_det_b_laue = operations_with_da.da_to_laue_hist(
                    da_det_b, factor_border=0.00
                )
                vmax = max([vmax, numpy.quantile(da_det_b_laue.data.values, 0.9)])
                d_plot["detector B"] = da_det_b_laue

            fig = plopp.scatter3d(
                d_plot,
                pos="event_position_global",
                cbar=True,
                size=0.005,
                opacity=0.75,
                vmax=vmax,
            )
        elif fig_rbuttons.value == "2D Pattern":
            data_event = STATE.detector_a_event
            if data_event is None:
                clear_output()
                return
            da_2d = operations_with_da.da_to_2d_hist(
                data_event, factor_border=0.07
            )
            STATE.data_event_hist = da_2d
            rad_to_deg = da_2d.assign_coords(
                gamma_event=da_2d.coords["event_gamma"].to(unit="deg"),
                nu_event=da_2d.coords["event_nu"].to(unit="deg"),
            )
            fig = plopp.inspector(
                rad_to_deg,
                dim="toa",
                orientation="vertical",
                logc=False,
                mode="rectangle",
                autoscale=True,
            )
        elif fig_rbuttons.value == "TOF-2Theta":
            data_event = STATE.detector_a_event
            if data_event is None:
                clear_output()
                return
            dg_magic = STATE.magic_data
            assign_dg_to_da_coords(dg_magic['sample'], data_event, prefix="sample")
            data_event.coords['tp_position'] = dg_magic['tp_position']
            data_event.coords['source_position'] = dg_magic['source_position']
            data_event.coords['delta_L'] = dg_magic['delta_L']
            data_event.coords['delta_t'] = dg_magic['delta_t']


            data_event = data_event.transform_coords(
            ("two_theta",), graph={**magic_graphs.graph_qvec, **magic_graphs.graph_detector}
            )
            rad_to_deg = data_event.assign_coords(
                two_theta=data_event.coords["two_theta"].to(unit="deg"),
            )
            rad_to_deg_hist = rad_to_deg.hist(two_theta=120, toa=1000)
            fig = plopp.plot(
                rad_to_deg_hist,
                coords=['toa', 'two_theta'],
            )
        elif fig_rbuttons.value == "3D Normalization Data":
            data_event = STATE_VANADIUM.detector_a_event
            if data_event is None:
                clear_output()
                return
            da_laue = operations_with_da.da_to_laue_hist(
                data_event, factor_border=0.07
            )
            vmax = numpy.quantile(da_laue.data.values, 0.9)
            fig = plopp.scatter3d(
                da_laue,
                pos="event_position_local",
                cbar=True,
                size=0.005,
                opacity=0.75,
                vmax=vmax,
            )
        elif fig_rbuttons.value == "2D Normalization Data":
            data_event = STATE_VANADIUM.detector_a_event
            if data_event is None:
                clear_output()
                return
            da_2d = operations_with_da.da_to_2d_hist(
                data_event, factor_border=0.07
            )
            STATE.data_event_hist_vanadium = da_2d
            rad_to_deg = da_2d.assign_coords(
                gamma_event=da_2d.coords["event_gamma"].to(unit="deg"),
                nu_event=da_2d.coords["event_nu"].to(unit="deg"),
            )
            fig = plopp.inspector(
                rad_to_deg,
                dim="toa",
                orientation="vertical",
                logc=False,
                mode="rectangle",
                autoscale=False,
            )
        elif fig_rbuttons.value == "Monitor Data":
            data_cave_monitor = STATE.monitor_cave
            if data_cave_monitor is None:
                clear_output()
                return
            fig = data_cave_monitor.hist(toa=101).plot()
        else:
            flag_display_center = False
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [4, 5, 6])

        clear_output()
        if flag_display_center:
            display_center(fig)
        else:
            display(fig)


def display_da_data(b):
    with output_data:
        clear_output()
        da = getattr(STATE, state_dropdown.value)
        display(da)


def run_peak_finder(b):
    with peaks_output:
        clear_output()
        display(spinner)

        indexation_output.clear_output()
        display_peaks_button.layout.display = "none"
        hide_peaks_button.layout.display = "none"

        if norm_rbuttons.value == "No Normalization":
            data_event = STATE.data_event  # b.data_event
        elif norm_rbuttons.value == "Per Monitor":
            if STATE.data_event_normalized_per_monitor is None:
                calc_normalization_per_monitor()
            data_event = (
                STATE.data_event_normalized_per_monitor
            )  # b.data_event
        elif norm_rbuttons.value == "Per Vanadium":
            if STATE.data_event_normalized_per_vanadium is None:
                calc_normalization_per_vanadium()
            data_event = (
                STATE.data_event_normalized_per_vanadium
            )  # b.data_event
        else:
            print("Something is wrong in run_peak_finder")
            return

        peaks = find_peaks(data_event)

        STATE.data_peaks = peaks
        clear_output()
        display_peaks(b)
        display_peaks_button.layout.display = "inline-block"
        hide_peaks_button.layout.display = "inline-block"


def run_peak_finder_hist(b):
    with peaks_output:
        clear_output()
        display(spinner)

        indexation_output.clear_output()
        display_peaks_button.layout.display = "none"
        hide_peaks_button.layout.display = "none"

        if norm_rbuttons.value == "No Normalization":
            da_det_a = STATE.detector_a_event  # b.data_event
        # elif norm_rbuttons.value == "Per Monitor":
        #     if STATE.data_event_normalized_per_monitor is None:
        #         calc_normalization_per_monitor()
        #     data_event = (
        #         STATE.data_event_normalized_per_monitor
        #     )  # b.data_event
        elif norm_rbuttons.value == "Per Vanadium":
            if STATE.detector_a_event_normalized is None:
                calc_normalization_per_vanadium()
            da_det_a = STATE.detector_a_event_normalized  # b.data_event
        else:
            print("Something is wrong in run_peak_finder_hist")
            return

        da_det_a = da_det_a.transform_coords(
            ("event_gamma", "event_nu", "event_position_global"),
            graph=magic_graphs.graph_detector,
            rename_dims=False,
        )
        da_hist_det_a = operations_with_da.da_to_2d_hist(da_det_a)
        STATE.detector_a_event_hist = da_hist_det_a

        np_toa, np_gamma, np_nu, sig_toa, sig_gamma, sig_nu = (
            operations_with_da.find_peaks_hist(da_hist_det_a, threshold=0.1)
        )
        range_sigma = 5
        operations_with_da.assign_event_peak_to_da(
            da_det_a,
            np_toa,
            np_gamma,
            np_nu,
            sig_toa,
            sig_gamma,
            sig_nu,
            range_sigma,
        )
        dg_magic = STATE.magic_data
        operations_with_da.assign_dg_to_da_coords(
            dg_magic["sample"], da_det_a, prefix="sample"
        )
        da_det_a.coords["tp_position"] = dg_magic["tp_position"]
        da_det_a.coords["source_position"] = dg_magic["source_position"]
        da_det_a.coords["delta_L"] = dg_magic["delta_L"]
        da_det_a.coords["delta_t"] = dg_magic["delta_t"]
        da_det_a = da_det_a.transform_coords(
            ("Q_vec_rot", "norm_Q", "two_theta"), graph=magic_graphs.graph_qvec
        )

        da_peaks = operations_with_da.calc_da_peaks_for_event_peak(da_det_a)
        STATE.da_peaks = da_peaks

        flag_peak = da_det_a.coords["event_peak"] != 0
        fig = plopp.scatter3d(
            da_det_a[flag_peak], pos="Q_vec_rot", size=0.005, perspective=False
        )

        clear_output()
        display_peaks_button.layout.display = "inline-block"
        hide_peaks_button.layout.display = "inline-block"
        display(fig)


def display_peaks(b):
    with peaks_output:
        clear_output()
        display(spinner)

        print("Peaks:")
        data_peaks = STATE.da_peaks
        np_q = data_peaks.coords["Q_vec_rot"].values

        d_in = {
            "Qx": np_q[:, 0],
            "Qy": np_q[:, 1],
            "Qz": np_q[:, 2],
            "Intensity": data_peaks.data.values,
        }
        flag_hkl = all(
            [hh in data_peaks.coords.keys() for hh in ("h", "k", "l")]
        )
        if flag_hkl:
            d_in["h"] = data_peaks.coords["h"].values
            d_in["k"] = data_peaks.coords["k"].values
            d_in["l"] = data_peaks.coords["l"].values

        pd_peaks = pd.DataFrame(d_in)
        fig = plopp.scatter3d(
            data_peaks, pos="Q_vec_rot", cbar=True, size=1, perspective=False
        )

        clear_output()
        display(pd_peaks, fig)


def indexate_peaks(b):
    with indexation_output:
        clear_output()
        display(spinner)

        da_peaks = STATE.da_peaks

        sc_cell_a = sc.scalar(cell_a.value, unit="angstrom")
        sc_cell_b = sc.scalar(cell_b.value, unit="angstrom")
        sc_cell_c = sc.scalar(cell_c.value, unit="angstrom")
        sc_cell_alpha = sc.scalar(cell_alpha.value, unit="deg")
        sc_cell_beta = sc.scalar(cell_beta.value, unit="deg")
        sc_cell_gamma = sc.scalar(cell_gamma.value, unit="deg")

        # First estimation
        euler_alpha = sc.scalar(1.0, unit="deg")
        euler_beta = sc.scalar(1.0, unit="deg")
        euler_gamma = sc.scalar(0.0, unit="deg")

        # Only strong peaks used for refinement
        factor = 0.3
        da_peaks_strong = da_peaks[
            da_peaks.data > factor * da_peaks.data.max()
        ]
        print("Number of stronger peaks", da_peaks_strong.size)

        print("# No refinement UB-matrix")
        ea_opt, sc_b_matrix, chi_sq = get_ub.get_euleur_opt(
            sc_cell_a,
            sc_cell_b,
            sc_cell_c,
            sc_cell_alpha,
            sc_cell_beta,
            sc_cell_gamma,
            da_peaks_strong.coords["Q_vec_rot"],
            da_peaks_strong.coords["sigma_Q_vec_rot"],
            euler_alpha,
            euler_beta,
            euler_gamma,
            graph_hkl=magic_graphs.graph_hkl,
            relfine_unit_cell=False,
            singony="cubic",
        )
        euler_alpha, euler_beta, euler_gamma = ea_opt[0], ea_opt[1], ea_opt[2]
        print(
            f"Optimized Euler angles (deg):\n {ea_opt[0].to(unit='deg').value:.2f} {ea_opt[1].to(unit='deg').value:.2f} {ea_opt[2].to(unit='deg').value:.2f}\n"
        )
        print(f"Chi-squared: {chi_sq:.4f}\n")

        print("# UB-matrix is refined")
        ea_opt, sc_b_matrix, chi_sq = get_ub.get_euleur_opt(
            sc_cell_a,
            sc_cell_b,
            sc_cell_c,
            sc_cell_alpha,
            sc_cell_beta,
            sc_cell_gamma,
            da_peaks_strong.coords["Q_vec_rot"],
            da_peaks_strong.coords["sigma_Q_vec_rot"],
            euler_alpha,
            euler_beta,
            euler_gamma,
            graph_hkl=magic_graphs.graph_hkl,
            relfine_unit_cell=True,
            singony="cubic",
        )

        print(
            f"Optimized Euler angles (deg):\n {ea_opt[0].to(unit='deg').value:.2f} {ea_opt[1].to(unit='deg').value:.2f} {ea_opt[2].to(unit='deg').value:.2f}\n"
        )
        print(f"Optimized B matrix:\n{sc_b_matrix.values}\n")
        unit_cell = magic_graphs.graph_ub_inv[
            (
                "cell_a",
                "cell_b",
                "cell_c",
                "cell_alpha",
                "cell_beta",
                "cell_gamma",
            )
        ](sc_b_matrix)
        ls_out = [
            "Optimized unit cell:",
        ]
        l_label = ["a", "b", "c", "alpha", "beta", "gamma"]
        for ind, label in enumerate(l_label):
            ls_out.append(
                f"{label:>10} : {unit_cell[ind].value:} {unit_cell[ind].unit:}"
            )
        print("\n".join(ls_out) + "\n")

        print(f"Chi-squared: {chi_sq:.4f}\n")

        apply_unit_cell_and_orientation_to_da(da_peaks, unit_cell, ea_opt)
        for name in [
            "detector_a_event",
            "detector_a_event_hist",
            "detector_a_event_normalized",
            "detector_a_event_hist_normalized",
            "detector_b_event",
            "detector_b_event_hist",
            "detector_b_event_normalized",
            "detector_b_event_hist_normalized",
        ]:
            if not getattr(STATE, name) is None:
                apply_unit_cell_and_orientation_to_da(
                    getattr(STATE, name), unit_cell, ea_opt
                )
        cell_a.value = unit_cell[0].value
        cell_b.value = unit_cell[1].value
        cell_c.value = unit_cell[2].value
        cell_alpha.value = unit_cell[3].value
        cell_beta.value = unit_cell[4].value
        cell_gamma.value = unit_cell[5].value
        STATE.da_peaks = da_peaks.transform_coords(
            ("h", "k", "l", "h_reduced", "k_reduced", "l_reduced"),
            graph=magic_graphs.graph_hkl,
        )
        clear_output()
        display(STATE.da_peaks)
        # display_peaks(b)


# EXTRA


def apply_unit_cell_and_orientation_to_da(da, unit_cell, euler_angles):
    da.coords["cell_a"] = unit_cell[0]
    da.coords["cell_b"] = unit_cell[1]
    da.coords["cell_c"] = unit_cell[2]
    da.coords["cell_alpha"] = unit_cell[3]
    da.coords["cell_beta"] = unit_cell[4]
    da.coords["cell_gamma"] = unit_cell[5]
    da.coords["euler_alpha"] = euler_angles[0]
    da.coords["euler_beta"] = euler_angles[1]
    da.coords["euler_gamma"] = euler_angles[2]
    magic_scipp.remove_coords_in_da(
        da,
        "h",
        "k",
        "l",
        "h_reduced",
        "k_reduced",
        "l_reduced",
        "u_matrix",
        "ub_matrix",
        "b_matrix",
    )


def unique_random_integers(N, m):
    max_size = N + 1
    size = min(m, max_size)
    return numpy.random.choice(max_size, size=size, replace=False)


def randomly_take_n_events(da, n):
    flag_border = sc.zeros(dims=da.dims, shape=da.shape, dtype=bool)
    for key in da.masks.keys():
        flag_border |= da.masks[key]
    da_not_border = da[~flag_border]
    np_arg = unique_random_integers(da_not_border.size - 1, n)
    return da_not_border[np_arg]


def find_peaks(data_event):
    """Find peaks by events"""
    data_event_random = randomly_take_n_events(data_event, 500000)

    data_event_random = data_event_random.transform_coords(
        ("Q_vec_rot",), graph=magic_graphs.graph_qvec
    )

    da_peaks = peak_find.find_multiple_peaks_accel(
        events_coords=data_event_random.coords["Q_vec_rot"],
        events_weight=data_event_random.data,
        merge_radius=0.1,
        basin_radius=0.2,
        max_seeds=5000,
        random_state=None,
        radius_factor=3.0,
    )

    return da_peaks


def display_center(*graphs):
    return display(
        widgets.VBox(graphs, layout=widgets.Layout(align_items="center"))
    )


def calc_normalization_per_monitor():
    data_event = STATE.data_event  # b.data_event
    data_cave_monitor = STATE.data_cave_monitor
    data_event_normalized = operations_with_da.normalize_per_cave_monitor(
        data_event, data_cave_monitor
    )
    STATE.data_event_normalized_per_monitor = data_event_normalized


def calc_normalization_per_vanadium():
    da_det_a = STATE.detector_a_event
    da_det_a_vanadium = STATE_VANADIUM.detector_a_event
    # da_det_a_normalized = (
    #     operations_with_da.normalize_da_event_by_vanadium_over_voxel(
    #         da_det_a, da_det_a_vanadium
    #     )
    # )
    da_det_a_normalized = (
        operations_with_da.normalize_da_event_by_vanadium_over_time(
            da_det_a, da_det_a_vanadium, factor=0.01
        )
    )
    STATE.detector_a_event_normalized = da_det_a_normalized


def run_peak_integration(b):
    with integration_output:
        clear_output()
        display(spinner)

        if norm_rbuttons.value == "No Normalization":
            da = STATE.detector_a_event  # b.data_event
        # elif norm_rbuttons.value == "Per Monitor":
        #     if STATE.data_event_normalized_per_monitor is None:
        #         calc_normalization_per_monitor()
        #     data_event = (
        #         STATE.data_event_normalized_per_monitor
        #     )  # b.data_event
        elif norm_rbuttons.value == "Per Vanadium":
            if STATE.detector_a_event_normalized is None:
                calc_normalization_per_vanadium()
            da = STATE.detector_a_event_normalized  # b.data_event
        else:
            print("Something is wrong in run_peak_finder_hist")
            return

        da.masks["detector_border"] = sc.zeros(
            dims=da.dims, shape=da.shape, dtype=bool
        )
        scale = 33.6992238296537
        integration_box = [0.5, 0.5, 0.5]

        dg_magic = STATE.magic_data
        operations_with_da.assign_dg_to_da_coords(
            dg_magic["sample"], da, prefix="sample"
        )
        da.coords["tp_position"] = dg_magic["tp_position"]
        da.coords["source_position"] = dg_magic["source_position"]
        da.coords["delta_L"] = dg_magic["delta_L"]
        da.coords["delta_t"] = dg_magic["delta_t"]

        np_hkl_int, np_fsq_exp, np_wavelength, np_tth = (
            integrate_peaks.naive_integration(da, integration_box, scale=scale)
        )

        # fig = plopp.scatter3d(
        #     da[::100], pos="hkl_vec", cbar=True, size=0.0001, opacity=0.5
        # )
        ls_out = [
            f"{hkl[0]:4} {hkl[1]:4} {hkl[2]:4} {fsq:10.2f} {wavelength:10.5f} {tth:7.2f}"
            for hkl, fsq, wavelength, tth in zip(
                np_hkl_int.transpose(),
                np_fsq_exp,
                np_wavelength[:, 0],
                np_tth[:, 0],
            )
        ]

        clear_output()
        # display(fig)
        print(
            "   H    K    L        Fsq Wavelength  2Theta\n"
            + "\n".join(ls_out)
        )
