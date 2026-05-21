import scipp as sc
import magic_graphs
import numpy

from scipy.ndimage import label, center_of_mass, sum as ndi_sum

def apply_detector_border(da, factor_border=0.07):
    if 'detector_border' in da.masks.keys():
        da.masks['detector_border'] |= da.coords["voxel_ID_VS_detector_a"] < 120*factor_border
    else:
        da.masks['detector_border'] = da.coords["voxel_ID_VS_detector_a"] < 120*factor_border
    da.masks['detector_border'] |= da.coords["voxel_ID_VS_detector_a"] > 120 * (1-factor_border)
    da.masks['detector_border'] |= da.coords["voxel_ID_a_detector_a"] < 128 * factor_border
    da.masks['detector_border'] |= da.coords["voxel_ID_a_detector_a"] > 128* (1-factor_border)

def da_to_laue_hist(da, factor_border:float = 0.07):
    data_laue = sc.groupby(da, 'voxel_ID_detector_a').sum('event')
    data_laue = data_laue.transform_coords(("detector_event_position_local",'position', 'voxel_ID_VS_detector_a', 'voxel_ID_a_detector_a', 'voxel_ID_c_detector_a'), graph=magic_graphs.graph_qvec, rename_dims=False)

    apply_detector_border(data_laue, factor_border=factor_border)
    return data_laue

def normalize_per_cave_monitor(da_q_event, da_cm, factor=0.1):
    da_cm = da_cm.transform_coords(("wavelength",), graph=magic_graphs.graph_cave_monitor, rename_dims=False)
    da_q_event = da_q_event.transform_coords(("wavelength",), graph=magic_graphs.graph_qvec, rename_dims=False)
        
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


def da_to_2d_hist(da, factor_border:float = 0.07):
    da = da.transform_coords(("gamma_event",'nu_event', 'voxel_ID_a_detector_a'), graph=magic_graphs.graph_qvec, rename_dims=False)
    delta_gamma_event = sc.scalar(0.15,unit='deg').to(unit='rad') 
    gamma_min = da.coords['gamma_event'].min()
    gamma_max = da.coords['gamma_event'].max()
    num_gamma = int(((gamma_max-gamma_min)/delta_gamma_event).value)
    bin_gamma = sc.linspace('gamma_event', gamma_min, gamma_max, num=num_gamma)

    delta_nu_event = sc.scalar(0.333,unit='deg').to(unit='rad') 
    nu_min = da.coords['nu_event'].min()
    nu_max = da.coords['nu_event'].max()
    num_nu = int(((nu_max-nu_min)/delta_nu_event).value)
    bin_nu = sc.linspace('nu_event', nu_min, nu_max, num=num_nu)

    delta_toa = sc.scalar(0.1e-3, unit='s')
    toa_min = da.coords['toa'].min()
    toa_max = da.coords['toa'].max()
    num_toa = int(((toa_max-toa_min)/delta_toa).value)
    bin_toa = sc.linspace('toa', toa_min, toa_max, num=num_toa)
    data_hist = da.hist(toa=bin_toa, nu_event=bin_nu, gamma_event=bin_gamma)
    return data_hist

def normalize_da_hist_by_vanadium(da_hist, da_hist_vanadium):
    np_w = da_hist_vanadium.sum('toa').values
    np_w/=np_w.max()
    np_w_unique = numpy.unique(np_w)
    np_w[np_w==0.] = np_w_unique[1]
    da_hist_norm = da_hist.copy()
    da_hist_norm.data.values /= np_w
    da_hist_norm.data.variances /= np_w


    np_w_time = da_hist_vanadium.sum(('nu_event','gamma_event')).values
    np_w_time /= np_w_time.max()
    factor = 0.03
    flag_time = np_w_time > factor * np_w_time.max()
    np_time = 0.5*(da_hist_vanadium.coords['toa'].values[:-1]+da_hist_vanadium.coords['toa'].values[1:])[flag_time]
    np_time_min = np_time.min()
    np_time_max = np_time.max()

    da_hist_norm = da_hist_norm['toa', sc.scalar(np_time_min, unit='s'):sc.scalar(np_time_max, unit='s')].copy()
    np_time_2 = 0.5*(da_hist_norm.coords['toa'].values[:-1]+da_hist_norm.coords['toa'].values[1:])
    np_w_time_2 = numpy.interp(np_time_2, np_time, np_w_time[flag_time])
    da_hist_norm.data.values /= numpy.expand_dims(np_w_time_2, axis=(1,2))
    da_hist_norm.data.variances /= numpy.expand_dims(np_w_time_2, axis=(1,2))

    return da_hist_norm

    

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