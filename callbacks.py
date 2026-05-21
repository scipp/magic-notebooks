from ui import *
from state import STATE
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

def load_file(b):
    with output_data:
        clear_output()
        peaks_output.clear_output()
        indexation_output.clear_output()
        display(spinner)

        STATE.clear()

        fig_rbuttons.layout.display = 'none'
        display_button.layout.display = 'none'
        state_dropdown.layout.display = 'none'
        display_da_button.layout.display = 'none'
        hide_output_data_button.layout.display = 'none'
        norm_rbuttons.layout.display = 'none'
        find_peaks_button.layout.display = 'none'  
        find_peaks_hist_button.layout.display = 'none'  
        cell_a.layout.display = 'none'
        cell_b.layout.display = 'none'
        cell_c.layout.display = 'none'
        cell_alpha.layout.display = 'none'
        cell_beta.layout.display = 'none'
        cell_gamma.layout.display = 'none'
        indexation_button.layout.display = 'none'

        file_path = fc.selected
        if file_path is None or not os.path.isfile(file_path):
            print("No file selected")
            return
        try:
            d_out = read_h5.read_h5_to_dict(file_path)
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        STATE.data_event = d_out['data_event']
        STATE.data_cave_monitor = d_out['data_cave_monitor']


        file_path = fc_vanadium.selected
        if not file_path is None and os.path.isfile(file_path):
            try:
                d_out = read_h5.read_h5_to_dict(file_path)
                STATE.data_event_vanadium = d_out['data_event']
                STATE.data_cave_monitor_vanadium = d_out['data_cave_monitor']
            except Exception as e:
                print(f"Error reading file: {e}")
        clear_output()
            
        display_data(b)
        fig_rbuttons.layout.display = 'inline-block'
        display_button.layout.display = 'inline-block'
        state_dropdown.layout.display = 'inline-block'
        display_da_button.layout.display = 'inline-block'
        hide_output_data_button.layout.display = 'inline-block'
        norm_rbuttons.layout.display = 'inline-block'
        find_peaks_button.layout.display = 'inline-block'
        find_peaks_hist_button.layout.display = 'inline-block'
        

def display_data(b):
    with output_data:
        clear_output()
        display(spinner)

        # Create plopp figure
        print(fig_rbuttons.value)
        flag_display_center = True
        if fig_rbuttons.value == '3D Laue Pattern':
            data_event = STATE.data_event
            da_laue = operations_with_da.da_to_laue_hist(data_event, factor_border=0.07)
            vmax = numpy.quantile(da_laue.data.values,0.9)
            fig = plopp.scatter3d(da_laue, pos='detector_event_position_local', cbar=True, size=0.005, opacity=0.75, vmax=vmax)
        elif fig_rbuttons.value == '2D Pattern':
            data_event = STATE.data_event
            da_2d = operations_with_da.da_to_2d_hist(data_event, factor_border=0.07)
            STATE.data_event_hist = da_2d
            rad_to_deg = da_2d.assign_coords(gamma_event=da_2d.coords["gamma_event"].to(unit="deg"), nu_event=da_2d.coords["nu_event"].to(unit="deg"))
            fig = plopp.inspector(rad_to_deg, dim='toa', orientation='vertical', logc=False, mode='rectangle', autoscale=True)
        elif fig_rbuttons.value == '3D Normalization Data':
            data_event = STATE.data_event_vanadium
            da_laue = operations_with_da.da_to_laue_hist(data_event, factor_border=0.07)
            vmax = numpy.quantile(da_laue.data.values,0.9)
            fig = plopp.scatter3d(da_laue, pos='detector_event_position_local', cbar=True, size=0.005, opacity=0.75, vmax=vmax)
        elif fig_rbuttons.value == '2D Normalization Data':
            data_event = STATE.data_event_vanadium
            da_2d = operations_with_da.da_to_2d_hist(data_event, factor_border=0.07)
            STATE.data_event_hist_vanadium = da_2d
            rad_to_deg = da_2d.assign_coords(gamma_event=da_2d.coords["gamma_event"].to(unit="deg"), nu_event=da_2d.coords["nu_event"].to(unit="deg"))
            fig = plopp.inspector(rad_to_deg, dim='toa', orientation='vertical', logc=False, mode='rectangle', autoscale=False)
        elif fig_rbuttons.value == 'Monitor Data':
            data_cave_monitor = STATE.data_cave_monitor
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
        cell_a.layout.display = 'none' 
        cell_b.layout.display = 'none' 
        cell_c.layout.display = 'none' 
        cell_alpha.layout.display = 'none' 
        cell_beta.layout.display = 'none' 
        cell_gamma.layout.display = 'none' 
        indexation_button.layout.display = 'none' 
        display_peaks_button.layout.display = 'none'
        hide_peaks_button.layout.display = 'none'

        
        if norm_rbuttons.value == "No Normalization":
            data_event = STATE.data_event # b.data_event
        elif norm_rbuttons.value == "Per Monitor":
            if STATE.data_event_normalized_per_monitor is None:
                calc_normalization_per_monitor()
            data_event = STATE.data_event_normalized_per_monitor # b.data_event
        elif norm_rbuttons.value == "Per Vanadium":
            if STATE.data_event_normalized_per_vanadium is None:
                calc_normalization_per_vanadium()
            data_event = STATE.data_event_normalized_per_vanadium # b.data_event
        else:
            print("Something is wrong in run_peak_finder")
            return
        
        peaks = find_peaks(data_event)

        STATE.data_peaks = peaks
        clear_output()
        display_peaks(b)
        cell_a.layout.display = 'inline-block' 
        cell_b.layout.display = 'inline-block' 
        cell_c.layout.display = 'inline-block' 
        cell_alpha.layout.display = 'inline-block' 
        cell_beta.layout.display = 'inline-block' 
        cell_gamma.layout.display = 'inline-block' 
        indexation_button.layout.display = 'inline-block' 
        display_peaks_button.layout.display = 'inline-block'
        hide_peaks_button.layout.display = 'inline-block'


def run_peak_finder_hist(b):
    with peaks_output:
        clear_output()
        display(spinner)

        indexation_output.clear_output()
        cell_a.layout.display = 'none' 
        cell_b.layout.display = 'none' 
        cell_c.layout.display = 'none' 
        cell_alpha.layout.display = 'none' 
        cell_beta.layout.display = 'none' 
        cell_gamma.layout.display = 'none' 
        indexation_button.layout.display = 'none' 
        display_peaks_button.layout.display = 'none'
        hide_peaks_button.layout.display = 'none'

        
        if norm_rbuttons.value == "No Normalization":
            data_event = STATE.data_event # b.data_event
        elif norm_rbuttons.value == "Per Monitor":
            if STATE.data_event_normalized_per_monitor is None:
                calc_normalization_per_monitor()
            data_event = STATE.data_event_normalized_per_monitor # b.data_event
        elif norm_rbuttons.value == "Per Vanadium":
            if STATE.data_event_normalized_per_vanadium is None:
                calc_normalization_per_vanadium()
            data_event = STATE.data_event_normalized_per_vanadium # b.data_event
        else:
            print("Something is wrong in run_peak_finder_hist")
            return
        
        peak_labels, peak_number = find_peaks_hist(STATE.data_event_hist)
    
        np_data = STATE.data_event_hist.values
        maskAll = numpy.ones(np_data.transpose().shape, dtype=bool)
        masks = {'All': maskAll}
        for i in range(1, peak_number+1):
            masks[f"Peak {i}"]=(peak_labels == i).transpose()
    
        data3d = np_data.transpose()        # shape (Z, Y, X)
        base_img = data3d.sum(axis=-1)

        mask_list = widgets.Select(
            options=list(masks.keys()),
            description="Masks",
            layout=widgets.Layout(width="200px", height="200px")
        )
        clear_output()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(base_img, cmap="Purples", aspect="auto")
        ax.set_title("Projection (sum over last axis)")
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()

        def update(change):
            name = change["new"]
            mask = masks[name]

            masked_img = (data3d * mask).sum(axis=-1)
            im.set_data(masked_img)
            im.set_clim(vmin=masked_img.min(), vmax=masked_img.max())

            ax.set_title(f"Masked: {name} {x:.1f} {y:.1f} {z:.1f}")

            fig.canvas.draw_idle()

        mask_list.observe(update, names="value")

        plt.show()
        display(mask_list)


def display_peaks(b):
    with peaks_output:
        clear_output()
        display(spinner)

        print("Peaks:")
        data_peaks = STATE.data_peaks
        np_q = data_peaks.coords['Q_vec_rot'].values
        d_in = {
            "Qx": np_q[:,0],
            "Qy": np_q[:,1],
            "Qz": np_q[:,2],
            "Intensity": data_peaks.data.values
            }
        flag_hkl =  all([hh in data_peaks.coords.keys() for hh in ('h','k','l')]) 
        if flag_hkl:
            d_in['h']=data_peaks.coords['h'].values
            d_in['k']=data_peaks.coords['k'].values
            d_in['l']=data_peaks.coords['l'].values
            
        pd_peaks = pd.DataFrame(d_in)
        fig = plopp.scatter3d(data_peaks, pos='Q_vec_rot', cbar=True, size=1, perspective=False)
        
        clear_output()
        display(pd_peaks, fig)

def indexate_peaks(b):
    with indexation_output:
        clear_output()
        display(spinner)

        da_peaks = STATE.data_peaks
        
        sc_cell_a = sc.scalar(cell_a.value, unit="angstrom")
        sc_cell_b = sc.scalar(cell_b.value, unit="angstrom")
        sc_cell_c = sc.scalar(cell_c.value, unit="angstrom")
        sc_cell_alpha = sc.scalar(cell_alpha.value, unit="deg")
        sc_cell_beta = sc.scalar(cell_beta.value, unit="deg")
        sc_cell_gamma = sc.scalar(cell_gamma.value, unit="deg")

        # First estimation
        euler_alpha = sc.scalar(1., unit="deg")
        euler_beta = sc.scalar(1., unit="deg")
        euler_gamma = sc.scalar(0., unit="deg")

        # Only strong peaks used for refinement
        factor = 0.3
        da_peaks_strong = da_peaks[da_peaks.data > factor* da_peaks.data.max()]
        print("Number of stronger peaks", da_peaks_strong.size)

        print("# No refinement UB-matrix")
        ea_opt, sc_b_matrix, chi_sq = get_ub.get_euleur_opt(
            sc_cell_a, sc_cell_b, sc_cell_c, sc_cell_alpha, sc_cell_beta, sc_cell_gamma, 
            da_peaks_strong.coords["Q_vec_rot"], da_peaks_strong.data.values,
            euler_alpha, euler_beta, euler_gamma, graph_hkl=magic_graphs.graph_hkl,
            relfine_unit_cell=False, singony='cubic')
        euler_alpha, euler_beta, euler_gamma = ea_opt[0],ea_opt[1],ea_opt[2]
        print(f"Optimized Euler angles (deg):\n {ea_opt[0].to(unit='deg').value:.2f} {ea_opt[1].to(unit='deg').value:.2f} {ea_opt[2].to(unit='deg').value:.2f}\n")
        print(f"Chi-squared: {chi_sq:.4f}\n")
    
        print("# UB-matrix is refined")
        ea_opt, sc_b_matrix, chi_sq = get_ub.get_euleur_opt(
            sc_cell_a, sc_cell_b, sc_cell_c, sc_cell_alpha, sc_cell_beta, sc_cell_gamma, 
            da_peaks_strong.coords["Q_vec_rot"], da_peaks_strong.data.values,
            euler_alpha, euler_beta, euler_gamma, graph_hkl=magic_graphs.graph_hkl,
            relfine_unit_cell=True, singony='cubic')

        print(f"Optimized Euler angles (deg):\n {ea_opt[0].to(unit='deg').value:.2f} {ea_opt[1].to(unit='deg').value:.2f} {ea_opt[2].to(unit='deg').value:.2f}\n")
        print(f"Optimized B matrix:\n{sc_b_matrix.values}\n")
        unit_cell = magic_graphs.graph_ub_inv[("cell_a", "cell_b", "cell_c", "cell_alpha", "cell_beta", "cell_gamma")](sc_b_matrix)
        ls_out=["Optimized unit cell:",]
        l_label = ["a","b","c","alpha","beta","gamma"]
        for ind, label in enumerate(l_label):
            ls_out.append(f"{label:>10} : {unit_cell[ind].value:} {unit_cell[ind].unit:}")
        print("\n".join(ls_out)+"\n")

        print(f"Chi-squared: {chi_sq:.4f}\n")
        
        apply_unit_cell_and_orientation_to_da(da_peaks, unit_cell, ea_opt)
        for name in ['data_event', 'data_event_vanadium','data_event_normalized_per_monitor','data_event_normalized_per_monitor']:
            if not getattr(STATE, name) is None:
                apply_unit_cell_and_orientation_to_da(getattr(STATE, name), unit_cell, ea_opt)
        cell_a.value = unit_cell[0].value
        cell_b.value = unit_cell[1].value
        cell_c.value = unit_cell[2].value
        cell_alpha.value = unit_cell[3].value
        cell_beta.value = unit_cell[4].value
        cell_gamma.value = unit_cell[5].value
        STATE.data_peaks = da_peaks.transform_coords(("h","k","l","h_reduced","k_reduced","l_reduced"), graph=magic_graphs.graph_hkl)  
        clear_output()
        display_peaks(b)


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
    magic_scipp.remove_coords_in_da(da, "h", "k", "l", "h_reduced", "k_reduced", "l_reduced", "u_matrix", "ub_matrix", "b_matrix")


def unique_random_integers(N, m):
    max_size = N + 1
    size = min(m, max_size)
    return numpy.random.choice(max_size, size=size, replace=False)

def randomly_take_n_events(da, n):
    flag_border = sc.zeros(dims=da.dims, shape= da.shape, dtype=bool)
    for key in da.masks.keys():
        flag_border |= da.masks[key]
    da_not_border = da[~flag_border]    
    np_arg = unique_random_integers(da_not_border.size-1, n)
    return da_not_border[np_arg]
    

def find_peaks(data_event):
    """Find peaks by events"""
    data_event_random = randomly_take_n_events(data_event, 500000)
    
    data_event_random = data_event_random.transform_coords(("Q_vec_rot",), graph=magic_graphs.graph_qvec)
    
    da_peaks = peak_find.find_multiple_peaks_accel(
    events_coords=data_event_random.coords['Q_vec_rot'],
    events_weight=data_event_random.data,
    merge_radius=0.1,
    basin_radius=0.2,
    max_seeds=5000,
    random_state=None,
    radius_factor=3.0,
    )

    return da_peaks


def find_peaks_hist(data_event_hist):
    """Find peaks by events"""
    np_data = data_event_hist.values
    np_flag_peaks = np_data > 0.1*(np_data.max()-np_data.min())+np_data.min()
    peak_labels, peak_number = label(np_flag_peaks)
    print(f"Number of peaks is {peak_number}")
    peak_centers = center_of_mass(np_data, peak_labels, range(1, peak_number+1))

    # bin_gamma = data_event_hist.coords['gamma_event']
    # ind_gamma_centers = [hh[2] for hh in peak_centers]
    # gamma_centers = numpy.interp(ind_gamma_centers, range(bin_gamma.size), bin_gamma.values)

    # bin_nu = data_event_hist.coords['nu_event']
    # ind_nu_centers = [hh[1] for hh in peak_centers]
    # nu_centers = numpy.interp(ind_nu_centers, range(bin_nu.size), bin_nu.values)

    # bin_toa = data_event_hist.coords['toa']
    # ind_toa_centers = [hh[0] for hh in peak_centers]
    # toa_centers = numpy.interp(ind_toa_centers, range(bin_toa.size), bin_toa.values)
    
    return peak_labels, peak_number
    
    
def display_center(*graphs):
    return display(widgets.VBox(graphs, layout=widgets.Layout(align_items='center')))




def calc_normalization_per_monitor():
    data_event = STATE.data_event # b.data_event
    data_cave_monitor = STATE.data_cave_monitor
    data_event_normalized = operations_with_da.normalize_per_cave_monitor(data_event, data_cave_monitor)
    STATE.data_event_normalized_per_monitor = data_event_normalized

def calc_normalization_per_vanadium():
    print("Normalization per vanadium is not currently implemented")
    data_event = STATE.data_event
    data_event_vanadium = STATE.data_event_vanadium
    # data_event_normalized = operations_with_da.normalize_per_vanadium(data_event, data_event_vanadium)
    STATE.data_event_normalized_per_vanadium = data_event
