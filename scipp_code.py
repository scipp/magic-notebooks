#!/usr/bin/env python

# # Single Crystal Data Reduction workflow for MAGiC diffractometer

import os
import pathlib
import numpy 
from numpy.lib import recfunctions as rfn
import scipy
from scipy.ndimage import label, center_of_mass, sum as ndi_sum

import matplotlib.pyplot as plt

import scipp as sc
from scippneutron.conversion import graph

import inspect
import plopp

import read_h5
import plot_data_ddict
import magic_graphs
import magic_scipp
import peak_find
import get_ub
import integrate_peaks
import operations_with_da

import np_cryst_functions

from resolver import ParameterResolver

import importlib


import mridul_position

# Preprocessing

# Load Event Data

# The data processing pipeline is currently designed for simulated data.  

# Uncomment the cell below to run a McStas simulation using the provided [McStas model](https://git.esss.dk/dmsc-instrumentmodels/magic). This will take some time.

# The simulation file contains information about event data on the detector and information from the cave monitor
import scippnexus
dg_nexus = scippnexus.load('mccode.nxs')
detector_a = dg_nexus['entry']['instrument']['magic_detector_a']
event_ids = detector_a['magic_detector_a_event_data'].coords['detector_number'].values
mridul_position.event_positions(dg_nexus, event_ids)

vertices = detector_a["pixel_shape"]["vertices"].to(unit="m")
detector_number_size = vertices.size//8
vertices_fold = vertices.fold(
            dim="vertex",
            sizes={"detector_number": detector_number_size, "faces": 4, 'side':2},
        )

vertices_lr = sc.mean(vertices_fold, dim='faces')
A = vertices_lr.values
idx = numpy.arange(A.shape[0]) % 2
result = A[numpy.arange(A.shape[0]), idx,:]
local_position = sc.vectors(dims=('detector_number',), values = result, unit='m')



transform = detector_a["depends_on"].compute()
transform = transform["time", 0].data



f_nexus_data = r"/Users/iuriikibalin/Documents/files/Areas/ESS/McStas_Simulation_MAGiC/sim_SC_test/mccode.h5"
dg_magic = read_h5.read_magic_from_nexus(f_nexus_data)

dg_sample = dg_magic['sample']
da_det_a = dg_magic['detector_a']
da_monitor = dg_magic['cave_monitor']

t_min = da_det_a.bins.coords['toa'].min()
t_max = da_det_a.bins.coords['toa'].max()
t_step = sc.scalar(unit=t_min.unit, value=1e-3)

bin_toa = operations_with_da.get_bin_by_step('toa', t_min.value, t_max.value, t_step.value, t_min.unit)
da_reduced = operations_with_da.get_da_reduced(da_det_a, bin_toa, count_min=0)

da_reduced = da_reduced.transform_coords(
    ("event_gamma",'event_r', 'event_position_global'),
    graph=magic_graphs.graph_detector,
    rename_dims=False
)

operations_with_da.move_data_from_dg_magic_to_da_reduced(dg_magic, da_reduced)
da_reduced = da_reduced.transform_coords(('norm_Q', 'Qx', ), graph=magic_graphs.graph_qvec, rename_dims=False)

g_min = da_reduced.coords['event_gamma'].min().value
g_max = da_reduced.coords['event_gamma'].max().value
bin_gamma = operations_with_da.get_bin_by_step('event_gamma', g_min, g_max, numpy.radians(0.3), da_reduced.coords['event_gamma'].unit)

n_min = da_reduced.coords['event_nu'].min().value
n_max = da_reduced.coords['event_nu'].max().value
bin_nu = operations_with_da.get_bin_by_step('event_nu', n_min, n_max, numpy.radians(0.3), da_reduced.coords['event_nu'].unit)

da_hist_tgn = da_reduced.hist(
    toa=bin_toa,
    event_gamma=bin_gamma,
    event_nu=bin_nu,
)

peaks_tgn = operations_with_da.find_peaks_hist(da_hist_tgn, threshold=0.1, flag_variance=True, binary_dilation=2)



events_xyz = numpy.array([
    da_reduced.coords['event_gamma'].values, 
    da_reduced.coords['event_nu'].values, 
    da_reduced.coords['toa'].values,
], dtype=float)

events_weight = da_reduced.data.values



peaks_xyz_sigma = numpy.array([
    peaks_tgn.coords['event_gamma_sigma'].values, 
    peaks_tgn.coords['event_nu_sigma'].values, 
    peaks_tgn.coords['toa_sigma'].values,
], dtype=float)

box_half_sizes = 3. * peaks_xyz_sigma.mean(axis=1) / 2.
bg_inner_half_sizes = box_half_sizes
bg_outer_half_sizes = bg_inner_half_sizes + box_half_sizes * 0.5

m_ub = numpy.array([[1./2., 0, 0], [0, 1./4.5, 0], [0, 0, 1./10.]], dtype=float)
m_r = numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
lambda_min, lambda_max = 0.5, 3.5

importlib.reload(np_cryst_functions)

np_peak = np_cryst_functions.generate_peak_data(
    m_ub,
    m_r,
    lambda_min=lambda_min,
    lambda_max=lambda_max,
)

delta_l = da_reduced.coords['delta_L'].values
delta_t = da_reduced.coords['delta_t'].values 
l_total = da_reduced.coords['Ltotal'].values.mean()

toa = delta_t + 0.001 * (l_total-delta_l)/3.9556/np_peak['wavelength']

np_peak = rfn.append_fields(
    np_peak,
    names='toa',
    data=toa,
    dtypes='f8',
    usemask=False
)

peaks_xyz = numpy.stack([np_peak['gamma'], np_peak['nu'], np_peak['toa']], axis=0)


preassign_nonoverlapping, ls_problems, l_peak_problem_ind = \
    integrate_peaks.check_peak_background_overlaps(
        peaks_xyz.T,
        box_half_sizes,
        bg_inner_half_sizes,
        bg_outer_half_sizes,
    )

l_ind = sorted(set(range(peaks_xyz.shape[1])) - set(l_peak_problem_ind))

peaks_xyz = peaks_xyz[:, l_ind]
preassign_nonoverlapping = True

intensity, error = integrate_peaks.integrate_peaks_md_box_memorysafe(
    events_xyz.T,
    events_weight,
    peaks_xyz.T,
    box_half_sizes,
    bg_inner_half_sizes=bg_inner_half_sizes,
    bg_outer_half_sizes=bg_outer_half_sizes,
    metadata_mask=None,
    preassign_nonoverlapping=preassign_nonoverlapping,
)

np_peak_small = np_peak[l_ind]


np_peak_small = rfn.append_fields(
    np_peak_small,
    names='intensity',
    data=intensity,
    dtypes='f8',
    usemask=False
)

np_peak_small = rfn.append_fields(
    np_peak_small,
    names='intensity_sigma',
    data=error,
    dtypes='f8',
    usemask=False
)
l_ind_non_zero = []
for i_hh, hh in enumerate(np_peak_small):
    if hh['intensity'] > 0.:
        l_ind_non_zero.append(i_hh)

np_peak_int = np_peak_small[l_ind_non_zero]
for hh in np_peak_int:
    print(f"{hh['h']:4.0f} {hh['k']:4.0f} {hh['l']:4.0f} {hh['intensity']:10.2f} {hh['intensity_sigma']:10.2f}")

np_peak_small
print(intensity)
np_peak[l_ind]
np_peak[l_peak_problem_ind]

numpy.degrees(np_peak['gamma'])
    
np_peak.dtype.names
hkl = numpy.column_stack([np_peak['h'], np_peak['k'], np_peak['l']]).T
for h,k,l in np_peak[['h','k','l']]:
    print(h, k, l)

np_peak = rfn.append_fields(
    np_peak,
    names='nu_deg',
    data=numpy.rad2deg(np_peak['nu']),
    dtypes='f8',
    usemask=False
)


d_param = {
    'np_gamma': peaks_tgn.coords['event_gamma'].to(unit='rad').values,
    'np_nu': peaks_tgn.coords['event_nu'].to(unit='rad').values,
    'np_toa': peaks_tgn.coords['toa'].to(unit='ms').values,
}

d_param['np_r'] = numpy.ones_like(d_param['np_gamma'])

def calc_detector_event_position_global(detector_position, detector_event_position):
    detector_event_position_global = numpy.expand_dims(detector_position, axis=1) + detector_event_position
    return detector_event_position_global

d_param['detector_position'] = dg_magic['detector_a'].coords['position'].to(unit='m').value
d_param['sample_omega'] = dg_magic['sample']['omega'].to(unit='rad').value
d_param['sample_chi'] = dg_magic['sample']['chi'].to(unit='rad').value
d_param['sample_phi'] = dg_magic['sample']['phi'].to(unit='rad').value
d_param['sample_position'] = dg_magic['sample']['position'].to(unit='m').value
d_param['source_position'] = dg_magic['source_position'].to(unit='m').value
d_param['tp_position'] = dg_magic['tp_position'].to(unit='m').value
d_param['delta_t'] = 3
d_param['delta_l'] = 0.1


np_cryst_functions.np_graph_qvec['event_position_global'] = calc_detector_event_position_global
np_cryst_functions.np_graph_qvec['detector_event_position'] = np_cryst_functions.calc_vector_by_gamma_nu_r

aliases = {
    'l_m':'l_total', 
    'gamma':'np_gamma', 
    'nu':'np_nu', 
    'r':'np_r', 
    'toa_ms':'np_toa',
    'delta_t_ms': 'delta_t',
}

d_param['delta_t'] = 30

def calc_chi_sq_lt(x):
    d_param['delta_t'] = x[0]
    d_param['delta_l'] = x[1]


    resolver1 = resolver.ParameterResolver(
        np_cryst_functions.np_graph_qvec, d_param, aliases
    )

    d_out = resolver1.resolve(('q_unrot', ), force_recompute=True, verbose=True)

    euler_angles = numpy.zeros((3,), dtype=float)
    unit_cell_parameters = numpy.array([2.890600,9.802400,12.580400,0.5*numpy.pi,0.5*numpy.pi,0.5*numpy.pi], dtype=float)


    ucp, ea, ub_matrix, res = get_ub.get_euler_opt_by_qvec(
        unit_cell_parameters, d_out['q_unrot'], 0.1, euler_angles, singony='ortho',
        refine_euler_angles=False, refine_unit_cell_parameters=False, 
        constraint_volume=True, minimization_basinhopping=False,
    )
    return res['fun']

x0 = [3, 0.1]

res = scipy.optimize.minimize(calc_chi_sq_lt, x0)
res

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_proj_type('ortho')
ax.set_xlabel('Q_X')
ax.set_ylabel('Q_Y')
ax.set_zlabel('Q_Z')
ax.scatter(d_out['q_unrot'][0],d_out['q_unrot'][1],d_out['q_unrot'][2])
ax.view_init(elev=90, azim=0) 
fig


print(ucp)
print(ea)
print(res)
np_hkl = numpy.linalg.inv(ub_matrix) @ d_out['q_unrot']

def func(x,y):
    return x + y


aa = numpy.array([1,3],dtype=float)
func(*aa)
q_unrot


plt.show()
resolver.build_dependency_graph()
resolver.visualize_dependency_graph()
sc.table(peaks_tgn)

np_graph_qvec.graph_qvec

sc.show_graph(magic_graphs.graph_qvec)

sc.show_graph(magic_graphs.graph_detector)


# plopp.scatter3d({'Detector A':da_det_a,'Detector B':da_det_b,}, pos='event_position_local', cbar=True, size=0.03, opacity=0.5, perspective=False)



# # Peak finder

# ## Data binning over $\gamma / \nu / $ time
# (in this case we solve the problem of long measurements when there is a lot of events)




da_det_a = da_det_a.transform_coords(("event_gamma",'event_nu', 'event_position_global'), graph=magic_graphs.graph_detector, rename_dims=False)
da_det_b = da_det_b.transform_coords(("event_gamma",'event_nu', 'event_position_global'), graph=magic_graphs.graph_detector, rename_dims=False)





da_det_a_vanadium = da_det_a_vanadium.transform_coords(("event_gamma",'event_nu', 'event_position_global'), graph=magic_graphs.graph_detector, rename_dims=False)
da_det_b_vanadium = da_det_b_vanadium.transform_coords(("event_gamma",'event_nu', 'event_position_global'), graph=magic_graphs.graph_detector, rename_dims=False)





da_hist_det_a = operations_with_da.da_to_2d_hist(da_det_a)
da_hist_det_b = operations_with_da.da_to_2d_hist(da_det_b)





da_hist_det_a_vanadium = operations_with_da.da_to_2d_hist(da_det_a_vanadium)
da_hist_det_b_vanadium = operations_with_da.da_to_2d_hist(da_det_b_vanadium)





# plopp.inspector(da_hist_det_a, dim='toa', orientation='vertical', logc=False, mode='rectangle')
# plopp.inspector(da_hist_det_b, dim='toa', orientation='vertical', logc=False, mode='rectangle')
# plopp.inspector(da_hist_det_b_vanadium, dim='toa', orientation='vertical', logc=False, mode='rectangle')


# ### Normalization of histogramm over  $\gamma / \nu / $ and over time




# da_hist_det_a_norm = operations_with_da.normalize_da_hist_by_vanadium(da_hist_det_a, da_hist_det_a_vanadium, factor_time=0.01)
da_hist_det_a_norm = operations_with_da.normalize_da_hist_by_vanadium_over_time(da_hist_det_a, da_hist_det_a_vanadium, factor=0.01)





# plopp.inspector(da_hist_det_a_norm, dim='toa', orientation='vertical', logc=False, mode='rectangle')


# ## Searching   $\gamma / \nu / $ time of strong peaks using histogrammed data
# 
# 




np_toa, np_gamma, np_nu, sig_toa, sig_gamma, sig_nu = operations_with_da.find_peaks_hist(da_hist_det_a, threshold=0.1)





range_sigma = 5
operations_with_da.assign_event_peak_to_da(da_det_a, np_toa, np_gamma, np_nu, sig_toa, sig_gamma, sig_nu, range_sigma)


# ### Peaks in Q-space




operations_with_da.assign_dg_to_da_coords(dg_magic['sample'], da_det_a, prefix="sample")
da_det_a.coords['tp_position'] = dg_magic['tp_position']
da_det_a.coords['source_position'] = dg_magic['source_position']
da_det_a.coords['delta_L'] = dg_magic['delta_L']
da_det_a.coords['delta_t'] = dg_magic['delta_t']
da_det_a = da_det_a.transform_coords(("Q_vec_rot","norm_Q", "two_theta"), graph=magic_graphs.graph_qvec)





da_peaks = operations_with_da.calc_da_peaks_for_event_peak(da_det_a)
da_peaks





plopp.scatter3d(da_peaks, pos='Q_vec_rot', size=1, perspective=False)





flag_peak = da_det_a.coords['event_peak'] != 0
plopp.scatter3d(da_det_a[flag_peak], pos='Q_vec_rot', size=0.005, perspective=False)


# # Indexing
# ## UB matrix based on the strong peaks using provided unit cell parameters










# Given by User
cell_a = sc.scalar(14.04078, unit="angstrom")
cell_b = sc.scalar(14.04078, unit="angstrom")
cell_c = sc.scalar(14.04078, unit="angstrom")
cell_alpha = sc.scalar(90., unit="deg")
cell_beta = sc.scalar(90., unit="deg")
cell_gamma = sc.scalar(90., unit="deg")

# First estimation
euler_alpha = sc.scalar(1., unit="deg")
euler_beta = sc.scalar(1., unit="deg")
euler_gamma = sc.scalar(0., unit="deg")

# Only strong peaks used for refinement
factor = 0.01
da_peaks_strong = da_peaks[da_peaks.data > factor* da_peaks.data.max()]
da_peaks_strong











print("# No refinement UB-matrix")
ea_opt, sc_b_matrix, chi_sq = get_ub.get_euleur_opt(
    cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma, 
    da_peaks_strong.coords["Q_vec_rot"], da_peaks_strong.coords["sigma_Q_vec_rot"],
    euler_alpha, euler_beta, euler_gamma, graph_hkl=magic_graphs.graph_hkl,
    relfine_unit_cell=False, singony='cubic')
euler_alpha, euler_beta, euler_gamma = ea_opt[0],ea_opt[1],ea_opt[2]
print(f"Optimized Euler angles (deg):\n {ea_opt[0].to(unit='deg').value:.2f} {ea_opt[1].to(unit='deg').value:.2f} {ea_opt[2].to(unit='deg').value:.2f}\n")
print(f"Chi-squared: {chi_sq:.4f}\n")

print("# UB-matrix is refined")
ea_opt, sc_b_matrix, chi_sq = get_ub.get_euleur_opt(
    cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma, 
    da_peaks_strong.coords["Q_vec_rot"], da_peaks_strong.coords["sigma_Q_vec_rot"],
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





threshold_strong = 0.01





da_peaks.coords["cell_a"] = unit_cell[0]
da_peaks.coords["cell_b"] = unit_cell[1]
da_peaks.coords["cell_c"] = unit_cell[2]
da_peaks.coords["cell_alpha"] = unit_cell[3]
da_peaks.coords["cell_beta"] = unit_cell[4]
da_peaks.coords["cell_gamma"] = unit_cell[5]
da_peaks.coords["euler_alpha"] = ea_opt[0]
da_peaks.coords["euler_beta"] = ea_opt[1]
da_peaks.coords["euler_gamma"] = ea_opt[2]
magic_scipp.remove_coords_in_da(da_peaks, "h", "k", "l", "h_reduced", "k_reduced", "l_reduced", "u_matrix", "ub_matrix", "b_matrix")

da_peaks = da_peaks.transform_coords(("h","k","l","h_reduced","k_reduced","l_reduced"), graph=magic_graphs.graph_hkl)  
da_peaks_strong = da_peaks[da_peaks.data > threshold_strong * da_peaks.data.max()]





plt.scatter(da_peaks_strong.coords["h"].values, da_peaks_strong.coords["k"].values)
plt.show()











da_det_a.coords["cell_a"] = unit_cell[0]
da_det_a.coords["cell_b"] = unit_cell[1]
da_det_a.coords["cell_c"] = unit_cell[2]
da_det_a.coords["cell_alpha"] = unit_cell[3]
da_det_a.coords["cell_beta"] = unit_cell[4]
da_det_a.coords["cell_gamma"] = unit_cell[5]
da_det_a.coords["euler_alpha"] = da_peaks.coords["euler_alpha"]
da_det_a.coords["euler_beta"] = da_peaks.coords["euler_beta"]
da_det_a.coords["euler_gamma"] = da_peaks.coords["euler_gamma"]
magic_scipp.remove_coords_in_da(da_det_a, "h", "k", "l", "h_reduced", "k_reduced", "l_reduced", "u_matrix", "ub_matrix", "b_matrix")
da_det_a = da_det_a.transform_coords(("h","k","l","h_reduced","k_reduced","l_reduced", "norm_Q"), graph={**magic_graphs.graph_hkl,**magic_graphs.graph_qvec})  


# ## Visualisation to see the quality of indexing




bin_h = sc.linspace(dim='h', start =-14,stop=-1,num=181,endpoint=True)
bin_k = sc.linspace(dim='k', start =-5,stop=8,num=341,endpoint=True)
bin_l = sc.linspace(dim='l', start =0,stop=10,num=341,endpoint=True)
# da_q_event_normalized.hist(h=bin_h).plot()
# da_q_event_normalized.hist(k=bin_k).plot()
# da_q_event_normalized.hist(l=bin_l).plot()





da_det_a.coords["h"] = da_det_a.coords["h"].copy()
da_det_a.coords["k"] = da_det_a.coords["k"].copy()
da_det_a.coords["l"] = da_det_a.coords["l"].copy()
da_det_a.coords["h_reduced"] = da_det_a.coords["h_reduced"].copy()
da_det_a.coords["k_reduced"] = da_det_a.coords["k_reduced"].copy()
da_det_a.coords["l_reduced"] = da_det_a.coords["l_reduced"].copy()





da_det_a.hist(h=bin_h, k=bin_k).plot(cmap='twilight')# norm="log"
da_det_a.hist(k=bin_k, l=bin_l).plot(cmap='twilight')# norm="log"
da_det_a.hist(h=bin_h, l=bin_l).plot(cmap='twilight')# norm="log"





bin_h_reduced = sc.linspace(dim='h_reduced', start =-0.5,stop=0.5,num=101,endpoint=True)
bin_k_reduced = sc.linspace(dim='k_reduced', start =-0.5,stop=0.5,num=101,endpoint=True)
bin_l_reduced = sc.linspace(dim='l_reduced', start =-0.5,stop=0.5,num=101,endpoint=True)





da_det_a.hist(h_reduced=bin_h_reduced, k_reduced=bin_k_reduced).plot(norm="log", cmap='twilight')





da_det_a.hist(k_reduced=bin_k_reduced, l_reduced=bin_l_reduced).plot(norm="log", cmap='twilight')





da_det_a.hist(l_reduced=bin_l_reduced, h_reduced=bin_h_reduced).plot(norm="log", cmap='twilight')





bin_q = sc.linspace(dim='norm_Q', start =1,stop=8,num=501,unit="1/Angstrom",endpoint=True)
da_det_a.hist(norm_Q=bin_q).plot()


# ## Correction on $\Delta t$, $\Delta L$, and sample offset
# 
# Correction is based on the events around strong peaks.
# 




sc_flag = sc.zeros(dims=da_det_a.data.dims, shape = da_det_a.data.values.shape, dtype=bool) 
distance_treshold = sc.scalar(0.1, unit="1/Angstrom")
da_det_a = da_det_a.transform_coords(("Q_vec_rot",), graph=magic_graphs.graph_qvec)
qvec = da_det_a.coords["Q_vec_rot"]
N= da_peaks_strong.coords["Q_vec_rot"].size
i=0
for q in da_peaks_strong.coords["Q_vec_rot"]:
    print(f"{100*(i+1)/N:.2f}%", end="\r")
    i+=1
    flag = (sc.norm(qvec-q) < distance_treshold)
    sc_flag = sc.logical_or(sc_flag, flag)

da_det_a_strong = da_det_a[sc_flag]
da_det_a_strong





da_det_a_strong = da_det_a_strong.transform_coords(("h_reduced", "k_reduced", "l_reduced"), graph={**magic_graphs.graph_qvec, **magic_graphs.graph_hkl})
# da_q_event_normalized_strong.hist(h_reduced=bin_h_reduced, k_reduced=bin_k_reduced).plot(cmap='twilight',)#norm="log", 





# da_q_event_normalized_strong.hist(k_reduced=bin_k_reduced, l_reduced=bin_l_reduced).plot(cmap='twilight',)# ,norm="log" 





# da_q_event_normalized_strong.hist(l_reduced=bin_l_reduced, h_reduced=bin_h_reduced).plot(cmap='twilight')# norm="log", 











get_ub.optimize_delta_t_delta_l(da_det_a_strong)





da_det_a_strong = da_det_a_strong.transform_coords(("h_reduced", "k_reduced", "l_reduced"), graph={**magic_graphs.graph_qvec, **magic_graphs.graph_hkl})






# da_q_event_normalized_strong.hist(h_reduced=bin_h_reduced, k_reduced=bin_k_reduced).plot(cmap='twilight',)#norm="log", 





# da_q_event_normalized_strong.hist(k_reduced=bin_k_reduced, l_reduced=bin_l_reduced).plot(cmap='twilight',)#norm="log", 





# da_q_event_normalized_strong.hist(l_reduced=bin_l_reduced, h_reduced=bin_h_reduced).plot(cmap='twilight',)#norm="log", 





da_det_a.coords["delta_t"] = da_det_a_strong.coords["delta_t"]
da_det_a.coords["delta_L"] = da_det_a_strong.coords["delta_L"]
da_det_a.coords["cell_a"] = da_det_a_strong.coords["cell_a"]
da_det_a.coords["cell_b"] = da_det_a_strong.coords["cell_b"]
da_det_a.coords["cell_c"] = da_det_a_strong.coords["cell_c"]
da_det_a.coords["cell_alpha"] = da_det_a_strong.coords["cell_alpha"]
da_det_a.coords["cell_beta"] = da_det_a_strong.coords["cell_beta"]
da_det_a.coords["cell_gamma"] = da_det_a_strong.coords["cell_gamma"]
da_det_a.coords["euler_alpha"] = da_det_a_strong.coords["euler_alpha"]
da_det_a.coords["euler_beta"] = da_det_a_strong.coords["euler_beta"]
da_det_a.coords["euler_gamma"] = da_det_a_strong.coords["euler_gamma"]
magic_scipp.remove_coords_in_da(da_det_a, "h", "k", "l", "h_reduced", "k_reduced", "l_reduced", "hkl_vec","Q_vec_rot","Q_vec","Qx","Qy","Qz","wavelength", "tof", "Ltotal", "Q", "u_matrix", "b_matrix", "ub_matrix")
da_det_a = da_det_a.transform_coords(("h_reduced", "k_reduced", "l_reduced"), graph={**magic_graphs.graph_qvec, **magic_graphs.graph_hkl})
da_det_a





da_det_a.hist(h_reduced=bin_h_reduced, k_reduced=bin_k_reduced).plot(norm="log", cmap='twilight',)





da_det_a.hist(k_reduced=bin_k_reduced, l_reduced=bin_l_reduced).plot(norm="log", cmap='twilight',)





da_det_a.hist(l_reduced=bin_l_reduced, h_reduced=bin_h_reduced).plot(norm="log", cmap='twilight',)


# ## Shape of the diffraction peak
# ...

# # Normalization of events by Vanadium




da_det_a_normalized = operations_with_da.normalize_da_event_by_vanadium_over_voxel(da_det_a, da_det_a_vanadium)





da_det_a_normalized = operations_with_da.normalize_da_event_by_vanadium_over_time(da_det_a, da_det_a_vanadium, factor=0.01)
da_det_a_normalized


# # Peak integration
# 
# Model Fsq (used in McStas simulation)
# 
















# def load_fsq(f_name:str):
#     with open(f_name, 'r') as fid:
#         l_content = fid.readlines()
#     l_content = [hh for hh in l_content if not hh.startswith("#")]
#     l_hkl, l_fsq = [], []
#     for line in l_content:
#         l_hh = line.strip().split()
#         l_hkl.append((int(l_hh[0]), int(l_hh[1]), int(l_hh[2])))
#         l_fsq.append(l_hh[-1])
#     np_hkl = numpy.array(l_hkl, dtype=int).transpose()
#     np_fsq = numpy.array(l_fsq, dtype = float)
#     return np_hkl, np_fsq
# f_name_fsq = "C60_tetra.hkl"
# np_hkl_model, np_fsq_model = load_fsq(f_name_fsq)


# Naive integration




import importlib
importlib.reload(integrate_peaks)

















da = da_det_a_normalized
da.masks['detector_border'] = sc.zeros(dims=da.dims, shape= da.shape, dtype=bool)
scale = 33.6992238296537
integration_box = [0.5, 0.5, 0.5]
np_hkl_int, np_fsq_exp, np_wavelength, np_tth = integrate_peaks.naive_integration(da, integration_box, scale=scale)








# Form CIF object for experiment




plopp.scatter3d(da[::100], pos='hkl_vec', cbar=True, size=0.0001, opacity=0.5)


# # Output
# 
# ## CrysPY




np_wavelength[:,0]





ls_out = [ 
            f"{hkl[0]:4} {hkl[1]:4} {hkl[2]:4} {fsq:10.2f} {wavelength:10.5f} {tth:7.2f}"
            for hkl, fsq, wavelength, tth in zip(
                np_hkl_int.transpose(), np_fsq_exp, np_wavelength[:,0], np_tth[:,0]
            )
        ]
print("   H    K    L        Fsq Wavelength  2Theta\n" + "\n".join(ls_out))





import cryspy





l_item = []
fsq_exp_max = np_fsq_exp.max()
for hkl, fsq, wavelength in zip(np_hkl_int.T, np_fsq_exp, np_wavelength):
    if fsq < 0.01 * fsq_exp_max:
        continue
    sfsq = 0.01 * fsq_exp_max
    l_item.append(
        cryspy.DiffrnRefln(index_h=hkl[0], index_k=hkl[1], index_l=hkl[2], intensity=fsq, intensity_sigma=sfsq, wavelength=wavelength[0])
    )
l_diffrn_refln = cryspy.DiffrnReflnL()
l_diffrn_refln.items = l_item
np_ub = da.coords['ub_matrix'].values
difrn_orient_matrix = cryspy.DiffrnOrientMatrix(
    ub_11 = np_ub[0,0],
    ub_12 = np_ub[0,1],
    ub_13 = np_ub[0,2],
    ub_21 = np_ub[1,0],
    ub_22 = np_ub[1,1],
    ub_23 = np_ub[1,2],
    ub_31 = np_ub[2,0],
    ub_32 = np_ub[2,1],
    ub_33 = np_ub[2,2],
)
phase = cryspy.Phase(label="c60", scale=1.)
extinction = cryspy.Extinction(model="gauss", mosaicity=0., radius=0.)
setup = cryspy.Setup(field=0.)
difrn_orient_matrix.form_object()
exp_sc = cryspy.Diffrn(data_name = "exp1")
exp_sc.add_items([l_diffrn_refln, difrn_orient_matrix, phase, extinction, setup])





s_cif_c60 = """data_c60

_cell_length_a 14.152000
_cell_length_b 14.152000
_cell_length_c 14.152000
_cell_angle_alpha 90.000000
_cell_angle_beta 90.000000
_cell_angle_gamma 90.000000

_space_group_name_H-M_alt "P a -3"
_space_group_IT_coordinate_system_code 1

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_adp_type
_atom_site_U_iso_or_equiv
_atom_site_B_iso_or_equiv
_atom_site_multiplicity
_atom_site_Wyckoff_symbol
  C11   C   0.229400   0.967500   0.101000   1.0   Biso   0.0038   0.0   24   d
  C12   C   0.246700   0.946000   0.006100   1.0   Biso   0.0038   0.0   24   d
  C21   C   0.208100   0.064600   0.128900   1.0   Biso   0.0038   0.0   24   d
  C22   C   0.206600   0.859900   0.964000   1.0   Biso   0.0038   0.0   24   d
  C23   C   0.171000   0.903700   0.159000   1.0   Biso   0.0038   0.0   24   d
  C34   C   0.223600   0.112200   0.962900   1.0   Biso   0.0038   0.0   24   d
  C24   C   0.243900   0.019200   0.936400   1.0   Biso   0.0038   0.0   24   d
  C31   C   0.205300   0.134900   0.061600   1.0   Biso   0.0038   0.0   24   d
  C32   C   0.150300   0.798300   0.020200   1.0   Biso   0.0038   0.0   24   d
  C33   C   0.132300   0.820700   0.118600   1.0   Biso   0.0038   0.0   24   d

"""
rcif_obj = cryspy.str_to_globaln(s_cif_c60)
rcif_obj.add_items([exp_sc,])
rcif_obj





cryspy.rhochi_no_refinement(rcif_obj)
rcif_obj.plots()





rcif_obj.diffrn_exp1.phase.scale_refinement = True
cryspy.rhochi_rietveld_refinement(rcif_obj)
rcif_obj.plots()





with open("c60.rcif", "w") as fid:
    fid.write(rcif_obj.to_cif())


# 
# Export results in formats compatible with refinement tools
# ## EasyDiffraction
# ...
# ## FullProf 
# ...
# ## Mag2Pol
# ...
# 



















# 

# 




https://scipp.github.io/user-guide/groupby.html

Probably use it for integration: asign area of detectors for indexation and integrate further














































































# Synthetic example with 3 peaks
rng = numpy.random.default_rng(0)

centers = numpy.array([
    [0.2, 0.1, 0.0],
    [0.8, -0.2, 0.3],
    [-0.4, 0.5, -0.1]
])

q_events = []
weights = []

for c in centers:
    pts = c + 0.03 * rng.standard_normal((800, 3))
    q_events.append(pts)
    weights.append(rng.uniform(0.5, 1.5, size=800))

q_events = numpy.vstack(q_events)
weights = numpy.concatenate(weights)

peaks, intensities = find_multiple_peaks(q_events, weights, max_peaks=3)

print("Detected peaks:\n", peaks)
print("Intensities:\n", intensities)





Detected peaks:
 [[-0.40223781  0.50135126 -0.09777805]
 [ 0.19728945  0.10152996 -0.00128652]]
Intensities:
 [795.38879818 794.77672387]











import numpy as np
from scipy.spatial import KDTree

def reduce_kspace_events(
    k_points,
    weights,
    target_count=None,
    distance_threshold=None,
):
    """
    Reduce weighted events in k-space by parallel merging of nearest neighbors.

    Parameters
    ----------
    k_points : (N, d) array
        Event coordinates in k-space.
    weights : (N,) array
        Event weights.
    target_count : int or None
        Stop when number of events <= target_count.
    distance_threshold : float or None
        Stop when all nearest-neighbor distances exceed this value.

    Returns
    -------
    k_points : (M, d) array
        Reduced coordinates.
    weights : (M,) array
        Reduced weights.
    """

    k_list = [np.array(k) for k in k_points]
    w_list = [float(w) for w in weights]

    while True:

        N = len(k_list)
        if target_count is not None and N <= target_count:
            break

        # Build KDTree
        tree = KDTree(np.vstack(k_list))

        # Query nearest neighbor for each point
        dists, idxs = tree.query(np.vstack(k_list), k=2)
        nn_dist = dists[:, 1]
        nn_idx = idxs[:, 1]

        # Check distance threshold stopping condition
        if distance_threshold is not None:
            if np.all(nn_dist > distance_threshold):
                break

        # Build list of candidate pairs (i, j, dist)
        pairs = [(i, nn_idx[i], nn_dist[i]) for i in range(N)]
        # Sort by distance
        pairs.sort(key=lambda x: x[2])

        merged = set()
        new_k = []
        new_w = []

        # Parallel merging: take smallest non-overlapping pairs
        for i, j, d in pairs:
            if i in merged or j in merged:
                continue
            if distance_threshold is not None and d > distance_threshold:
                break

            # Merge i and j
            wi, wj = w_list[i], w_list[j]
            ki, kj = k_list[i], k_list[j]

            w_new = wi + wj
            k_new = (wi * ki + wj * kj) / w_new

            new_k.append(k_new)
            new_w.append(w_new)

            merged.add(i)
            merged.add(j)

        # Add all unmerged points
        for idx in range(N):
            if idx not in merged:
                new_k.append(k_list[idx])
                new_w.append(w_list[idx])

        k_list = new_k
        w_list = new_w

        # If no merges happened, stop
        if len(merged) == 0:
            break

    return np.vstack(k_list), np.array(w_list)







# Example: 2000 events in 3D
N = 2000000
k = np.random.randn(N, 1)
w = np.random.rand(N)

k_red, w_red = reduce_kspace_events(
    k, w,
    target_count=300,
    distance_threshold=0.05
)

print(k_red.shape, w_red.shape)





k_red.min(), k_red.max()


# 	◦ max_seeds: e.g. 10^3–10^4 seeds instead of all events.
# 	◦ radius_factor: trade off accuracy vs speed (smaller → fewer neighbors).
# 	◦ Region cuts in Q before building the tree.




k_red, w_red = reduce_kspace_events(
    k_red, w_red,
    target_count=300,
    distance_threshold=0.05
)

print(k_red.shape, w_red.shape)





peaks, intensities = find_multiple_peaks_accel(np_qxyz, np_weight, max_peaks=30)
print("Detected peaks:\n", peaks)
print("Intensities:\n", intensities)




















# # To delete

# ## Calibration: normalization per incident spectra estimated by cave monitor
# 
# The diffracted signal is normalised per incident spectrum estimated by a monitor in a cave, and the events with wavelength outside the specified minimal and maximal wavelength range are cut off.




da_cm = data_cave_monitor.transform_coords(("wavelength",), graph=magic_graphs.graph_cave_monitor)
da_cm
# Check overlapping wavelength





# TODO: It should be normalized per one and time 
# TODO: It should be an option load normalization from external file as monitor is not quite stable

def normalize_per_cave_monitor(da_q_event, da_cm, factor=0.1):
    da_cm.masks["counts"] = sc.logical_not(da_cm.data > factor*da_cm.data.max())
    cm_wavelength = da_cm.coords['wavelength'][sc.logical_not(da_cm.masks["counts"])]
    cm_weight = da_cm.data[sc.logical_not(da_cm.masks["counts"])]
    cm_weight = cm_weight/cm_weight.max().values
    cm_wavelength_min = cm_wavelength.min()
    cm_wavelength_max = cm_wavelength.max()
    print(f"Minimal wavelength is {cm_wavelength_min.value:7.5f} {cm_wavelength_min.unit}")
    print(f"Maximal wavelength is {cm_wavelength_max.value:7.5f} {cm_wavelength_max.unit}")

    flag = sc.logical_and(da_q_event.coords['wavelength'] >  cm_wavelength_min, da_q_event.coords['wavelength'] < cm_wavelength_max)
    da_q_event_reduced = da_q_event[flag]
    coeff = numpy.interp(da_q_event_reduced.coords['wavelength'].values, cm_wavelength.values, cm_weight.values)
    da_q_event_reduced.data  = da_q_event_reduced.data /sc.array(dims=("event",), values=coeff, unit=da_q_event_reduced.data.unit)
    return da_q_event_reduced





factor = 0.1
diff_time = da_q_event.coords['toa'].max() - da_q_event.coords['toa'].min()
da_q_event_normalized = normalize_per_cave_monitor(da_q_event, da_cm, factor=factor)
# da_q_event_normalized['detector_border'] |= da_q_event_normalized.coords['toa'] >  da_q_event_normalized.coords['toa'].max() - factor_border * diff_time
# da_q_event_normalized['detector_border'] |= da_q_event_normalized.coords['toa'] <  da_q_event_normalized.coords['toa'].min() + factor_border * diff_time
f1 = da_q_event_normalized.coords['toa'] >  da_q_event_normalized.coords['toa'].max() - factor_border * diff_time
f2 = da_q_event_normalized.coords['toa'] <  da_q_event_normalized.coords['toa'].min() + factor_border * diff_time
f_time = sc.logical_or(f1, f2)
da_q_event_normalized.masks['detector_border'] = sc.logical_or(da_q_event_normalized.masks['detector_border'], f_time)
da_q_event_normalized





plopp.scatter3d(da_q_event_normalized[~da_q_event_normalized.masks['detector_border']], pos='Q_vec_rot', cbar=True, size=0.03, opacity=0.5, perspective=False)





flag_peak = da_det_a.coords['event_peak']==1
plopp.scatter3d(da_det_a[flag_peak], pos='event_position_local', cbar=True, size=0.005, perspective=True) # opacity=0.5


# ## Signal from cave monitor
# 
# TODO: here the step in time is 1ms, modify it to 10 microseconds




da_monitor.hist(toa=101).plot()


# ## Laue pattern (integrated over time) on the detector A
# 
# TODO: I would like to merge events grouping them using ID of voxels and time (around 100 time points). It will give max around 5e7 weighted events that significantly less the number of raw measurements during one hour (around 1e9-1e11).

# Mask peaks on border detector




factor_border = 0.07
print(data_event.coords["voxel_ID_VS_detector_a"].min())
print(data_event.coords["voxel_ID_VS_detector_a"].max())
print(data_event.coords["voxel_ID_a_detector_a"].min())
print(data_event.coords["voxel_ID_a_detector_a"].max())
print(data_event.coords["voxel_ID_c_detector_a"].min())
print(data_event.coords["voxel_ID_c_detector_a"].max())

def apply_detector_border(da, factor_border=0.07):
    if 'detector_border' in da.masks.keys():
        da.masks['detector_border'] |= da.coords["voxel_ID_VS_detector_a"] < 120*factor_border
    else:
        da.masks['detector_border'] = da.coords["voxel_ID_VS_detector_a"] < 120*factor_border
    da.masks['detector_border'] |= da.coords["voxel_ID_VS_detector_a"] > 120 * (1-factor_border)
    da.masks['detector_border'] |= da.coords["voxel_ID_a_detector_a"] < 128 * factor_border
    da.masks['detector_border'] |= da.coords["voxel_ID_a_detector_a"] > 128* (1-factor_border)





data_laue = sc.groupby(da_det_a, 'voxel_ID').sum('event')
data_laue





data_laue = data_laue.transform_coords(("event_position_local",'event_position_global', 'voxel_ID_VS', 'voxel_ID_a', 'voxel_ID_c'), graph=magic_graphs.graph_detector, rename_dims=False)

# apply_detector_border(data_laue, factor_border=factor_border)
# apply_detector_border(data_event, factor_border=factor_border)

vmax = numpy.quantile(data_laue.data.values,0.9)
plopp.scatter3d(data_laue, pos='event_position_local', cbar=True, size=0.005, opacity=0.75, vmax=vmax)








# # Peak finding
# 
# The peaks are identified from event-mode data by applying statistical clustering techniques. Searching is performed for event normalized data.
# 
# **TODO:**
#  - Sigmas for found peaks in Q space
#  - merge radius should be dependent form the resolution of the diffractometer.
#  - Basin radius should be dependent from expected distances between two closest peaks.




data_event.size





data_event.masks





flag_not_border = ~da_q_event_normalized.masks['detector_border']
da_peaks = peak_find.find_multiple_peaks_accel(
    events_coords=da_q_event_normalized[flag_not_border][::100].coords['Q_vec_rot'],
    events_weight=da_q_event_normalized[flag_not_border][::100].data,
    merge_radius=0.1,
    basin_radius=0.2,
    max_seeds=5000,
    random_state=None,
    radius_factor=3.0,
)
da_peaks
# info like from detector


# ## Checking the found peaks




peak_index = 4
sc.norm(da_peaks.coords["Q_vec_rot"][peak_index])





da_q_event_normalized.coords['omega_vs_detector_a'].to(unit='deg')





sc.norm(da_peaks.coords["Q_vec_rot"][peak_index])





peak_index = 10
flag = sc.norm(da_q_event_normalized.coords["Q_vec_rot"] - da_peaks.coords["Q_vec_rot"][peak_index]) < sc.scalar(0.10, unit="1/Angstrom")
da_one_peak = da_q_event_normalized[flag]

plopp.scatter3d(da_one_peak, pos='Q_vec', cbar=True, size=0.003, opacity=0.8, perspective=False)





peak_index = 10
flag = sc.norm(da_q_event_normalized.coords["Q_vec_rot"] - da_peaks.coords["Q_vec_rot"][peak_index]) < sc.scalar(0.025, unit="1/Angstrom")
da_one_peak = da_q_event_normalized[flag]

flag2 = da_one_peak.coords['voxel_ID_a_detector_a']==82
da_one_peak = da_one_peak[flag2]

flag2 = da_one_peak.coords['voxel_ID_VS_detector_a']==62
da_one_peak = da_one_peak[flag2]

plopp.scatter3d(da_one_peak, pos='Q_vec', cbar=True, size=0.001, opacity=0.8, perspective=True)











print (da_one_peak.coords['voxel_ID_VS_detector_a'].min().value,da_one_peak.coords['voxel_ID_VS_detector_a'].max().value)
print (da_one_peak.coords['voxel_ID_a_detector_a'].min().value,da_one_peak.coords['voxel_ID_a_detector_a'].max().value)
print(da_one_peak.coords['voxel_ID_c_detector_a'].min().value,da_one_peak.coords['voxel_ID_c_detector_a'].max().value)


# Found peaks in reciprocal spaces
# 
# **TODO:**
#  - More intensive peaks have large size (shape should be sphere)
#  - Manual selection of peaks to use it for furher searching of UB-matrix




# TODO: Selecting Q planes and projection on it 
max_size = 2
plopp.scatter3d(da_peaks, pos='Q_vec_rot', cbar=True, size=0.1, opacity=0.5)





max_size = 30
plt.scatter(da_peaks.coords["Q_vec_rot"].values[:,1], da_peaks.coords["Q_vec_rot"].values[:,2], s=max_size*da_peaks.data.values/da_peaks.data.values.max())


# Taking only strong peaks
# 
# TOD: Thrreshold in intensity normalized per time and incident spectra




threshold_strong = 0.1
da_peaks_strong = da_peaks[da_peaks.data > threshold_strong* da_peaks.data.max()]
da_peaks_strong


# 













# 







# ## Estimation the size of the peaks in Q space: radial size and the transverse one
















ind_peak = 0
l_q_peak_std = []
N_peak = da_peaks.size
l_xy = []
for ind_peak in range(N_peak):
    print(f"Progress {100*(ind_peak+1)/N_peak:.2f}%",end="\r")
    q_peak = da_peaks.coords["Q_vec_rot"][ind_peak]
    flag_peak = sc.norm(da_q_event_normalized.coords['Q_vec_rot']-q_peak)<sc.scalar(0.1, unit="1/Angstrom")
    da = da_q_event_normalized[flag_peak]

    np_q_vec = da.coords['Q_vec_rot'].values


    np_q_diff = numpy.abs((da.coords['Q_vec_rot']-q_peak).values)
    np_weight = numpy.expand_dims(da.data.values,axis=1)
    weight_sum = np_weight.sum()
    e_u = (q_peak/sc.norm(q_peak)).values

    np_q_vec_para = (np_q_vec *e_u).sum(axis=1)
    np_q_vec_perp = numpy.linalg.norm( np_q_vec -numpy.expand_dims(np_q_vec_para,axis=1)*numpy.expand_dims(e_u, axis=0), axis=1)


    if weight_sum <= 0.:
        continue

    # Weighted mean along radial direction (should be ~0 if centered well)
    mu_par = numpy.sum(np_weight[:,0] * np_q_vec_para) / weight_sum
    var_par = numpy.sum(np_weight[:,0] * (np_q_vec_para - mu_par)**2) / weight_sum
    sigma_par = numpy.sqrt(var_par)

    # Weighted mean of transverse radius
    var_perp = numpy.sum(np_weight[:,0] * (np_q_vec_perp)**2) / weight_sum
    sigma_perp = numpy.sqrt(var_perp)

    np_wavelength = da.coords['wavelength'].values
    wavelength = numpy.sum(np_weight[:,0] * np_wavelength) / weight_sum

    tth = numpy.sum(np_weight[:,0] * da.coords['two_theta'].to(unit="deg").values) / weight_sum
    da_hist = da.hist(norm_Q=101)
    l_xy.append((da_hist.coords["norm_Q"].values[:-1], da_hist.data.values))

    l_q_peak_std.append((mu_par, sigma_par, sigma_perp, wavelength,tth))
np_peak_param = numpy.array(l_q_peak_std,dtype=float).transpose()





n_max = 200
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
ax.scatter(np_peak_param[0,:n_max], np_peak_param[1,:n_max])
ax.set_xlabel("Q (inversed angstrems)")
ax.set_ylabel("Delta Q (inversed angstrems)")
ax.set_title("Width of peak along the radial direction in Q")





fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
ax.scatter(np_peak_param[0,:n_max]*numpy.power(np_peak_param[3,:n_max],-4), np_peak_param[1,:n_max])
ax.set_xlabel("Q / wavelength^4")
ax.set_ylabel("delta Q (inversed angstrems)")
ax.set_title("Width of peak along the radial direction in Q")





fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
ax.scatter(np_peak_param[0,:n_max], np_peak_param[2,:n_max])
ax.set_xlabel("Q (inversed angstrems)")
ax.set_ylabel("Q (inversed angstrems)")
ax.set_title("Width of peak perpendicular to radial direction in Q")





# Look ion data with totally different wavelenth if wavelength dependence is here





np_xy = numpy.array(l_xy)
np_xy.shape
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
for i in range(120):
    ax.plot(np_xy[i,0,:],np_xy[i,1,:])
ax.set_title("Individual peaks along Q")
ax.set_xlabel("Q (inversed angstrems)")
ax.set_ylabel("Signal (arb.u.)")

