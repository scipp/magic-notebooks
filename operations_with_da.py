import numpy

import scipp as sc
import magic_graphs

from scipy.ndimage import (
    label,
    center_of_mass,
    binary_dilation,
    sum as ndi_sum,
)


def apply_detector_border(da, factor_border=0.07):
    if "detector_border" in da.masks.keys():
        da.masks["detector_border"] |= (
            da.coords["voxel_ID_VS"] < 2 * da.coords["N_vs"] * factor_border
        )
    else:
        da.masks["detector_border"] = (
            da.coords["voxel_ID_VS"] < 2 * da.coords["N_vs"] * factor_border
        )
    da.masks["detector_border"] |= da.coords["voxel_ID_VS"] > 2 * da.coords[
        "N_vs"
    ] * (1 - factor_border)
    da.masks["detector_border"] |= (
        da.coords["voxel_ID_a"] < da.coords["N_a"] * factor_border
    )
    da.masks["detector_border"] |= da.coords["voxel_ID_a"] > da.coords[
        "N_a"
    ] * (1 - factor_border)


def da_to_laue_hist(da, factor_border: float = 0.07):
    # data_laue = da.hist()
    data_laue = da.group('detector_number').hist()
    data_laue = data_laue.transform_coords(
        (
            "event_position_local",
            "event_position_global",
            "voxel_ID_VS",
            "voxel_ID_a",
            "voxel_ID_c",
        ),
        graph=magic_graphs.graph_detector,
        rename_dims=False,
    )

    apply_detector_border(data_laue, factor_border=factor_border)
    return data_laue


def da_to_2d_hist(da, factor_border: float = 0.07):
    da = da.transform_coords(
        ("event_gamma", "event_nu", "voxel_ID_a"),
        graph=magic_graphs.graph_detector,
        rename_dims=False,
    )
    delta_gamma_event = sc.scalar(0.15, unit="deg").to(unit="rad")
    gamma_min = da.coords["event_gamma"].min()
    gamma_max = da.coords["event_gamma"].max()
    num_gamma = int(((gamma_max - gamma_min) / delta_gamma_event).value)
    bin_gamma = sc.linspace("event_gamma", gamma_min, gamma_max, num=num_gamma)

    delta_nu_event = sc.scalar(0.333, unit="deg").to(unit="rad")
    nu_min = da.coords["event_nu"].min()
    nu_max = da.coords["event_nu"].max()
    num_nu = int(((nu_max - nu_min) / delta_nu_event).value)
    bin_nu = sc.linspace("event_nu", nu_min, nu_max, num=num_nu)

    delta_toa = sc.scalar(0.1e-3, unit="s")
    toa_min = da.coords["toa"].min()
    toa_max = da.coords["toa"].max()
    num_toa = int(((toa_max - toa_min) / delta_toa).value)
    bin_toa = sc.linspace("toa", toa_min, toa_max, num=num_toa)
    data_hist = da.hist(toa=bin_toa, event_nu=bin_nu, event_gamma=bin_gamma)
    return data_hist


def normalize_da_event_by_cave_monitor(da_q_event, da_cm, factor=0.1):
    da_cm = da_cm.transform_coords(
        ("wavelength",),
        graph=magic_graphs.graph_cave_monitor,
        rename_dims=False,
    )
    da_q_event = da_q_event.transform_coords(
        ("wavelength",), graph=magic_graphs.graph_qvec, rename_dims=False
    )

    da_cm.masks["counts"] = sc.logical_not(
        da_cm.data > factor * da_cm.data.max()
    )
    cm_wavelength = da_cm.coords["wavelength"][
        sc.logical_not(da_cm.masks["counts"])
    ]
    cm_weight = da_cm.data[sc.logical_not(da_cm.masks["counts"])]
    cm_weight = cm_weight / cm_weight.max().values
    cm_wavelength_min = cm_wavelength.min()
    cm_wavelength_max = cm_wavelength.max()
    print(
        f"Minimal wavelength is {cm_wavelength_min.value:7.5f} {cm_wavelength_min.unit}"
    )
    print(
        f"Maximal wavelength is {cm_wavelength_max.value:7.5f} {cm_wavelength_max.unit}"
    )

    flag = sc.logical_and(
        da_q_event.coords["wavelength"] > cm_wavelength_min,
        da_q_event.coords["wavelength"] < cm_wavelength_max,
    )
    da_q_event_reduced = da_q_event[flag]
    coeff = numpy.interp(
        da_q_event_reduced.coords["wavelength"].values,
        cm_wavelength.values,
        cm_weight.values,
    )
    da_q_event_reduced.data = da_q_event_reduced.data / sc.array(
        dims=("event",), values=coeff, unit=da_q_event_reduced.data.unit
    )
    return da_q_event_reduced


def normalize_da_event_by_vanadium_over_voxel(da_event, da_vanadium):
    da_vanadium_voxel = sc.groupby(da_vanadium, "voxel_ID").sum("event")
    normalized = da_event.group(
        da_vanadium_voxel.coords["voxel_ID"]
    ) / sc.values(da_vanadium_voxel)
    da_event_normalized = normalized.bins.concat().value.copy()
    for name, item in normalized.coords.items():
        if name != "voxel_ID":
            da_event_normalized.coords[name] = item.copy()
    return da_event_normalized


def normalize_da_event_by_vanadium_over_time(
    da_event, da_vanadium, factor: float = 0.03
):
    da_spectra = da_vanadium.hist(toa=1000)
    da_spectra.masks["counts"] = sc.logical_not(
        da_spectra.data > factor * da_spectra.data.max()
    )
    spectra_toa = sc.midpoints(da_spectra.coords["toa"])[
        sc.logical_not(da_spectra.masks["counts"])
    ]
    spectra_weight = da_spectra.data[
        sc.logical_not(da_spectra.masks["counts"])
    ]
    spectra_weight = spectra_weight / spectra_weight.max().values

    spectra_toa_min = spectra_toa.min()
    spectra_toa_max = spectra_toa.max()
    print(
        f"Minimal toa is {spectra_toa_min.value:7.5f} {spectra_toa_min.unit}"
    )
    print(
        f"Maximal toa is {spectra_toa_max.value:7.5f} {spectra_toa_max.unit}"
    )

    flag = sc.logical_and(
        da_event.coords["toa"] > spectra_toa_min,
        da_event.coords["toa"] < spectra_toa_max,
    )
    da_event_reduced = da_event[flag]
    coeff = numpy.interp(
        da_event_reduced.coords["toa"].values,
        spectra_toa.values,
        spectra_weight.values,
    )
    da_event_reduced.data = da_event_reduced.data / sc.array(
        dims=("event",), values=coeff, unit=da_event_reduced.data.unit
    )
    return da_event_reduced


def normalize_da_hist_by_vanadium_over_time(
    da_hist, da_hist_vanadium, factor: float = 0.03
):
    da_hist_norm = da_hist.copy()
    np_w_time = da_hist_vanadium.sum(("event_nu", "event_gamma")).values
    np_w_time /= np_w_time.max()
    flag_time = np_w_time > factor * np_w_time.max()

    np_time = (
        0.5
        * (
            da_hist_vanadium.coords["toa"].values[:-1]
            + da_hist_vanadium.coords["toa"].values[1:]
        )[flag_time]
    )
    np_time_min = np_time.min()
    np_time_max = np_time.max()
    da_hist_norm = da_hist_norm[
        "toa",
        sc.scalar(np_time_min, unit="s") : sc.scalar(np_time_max, unit="s"),
    ].copy()
    np_time_2 = 0.5 * (
        da_hist_norm.coords["toa"].values[:-1]
        + da_hist_norm.coords["toa"].values[1:]
    )
    np_w_time_2 = numpy.interp(np_time_2, np_time, np_w_time[flag_time])
    da_hist_norm.data.values /= numpy.expand_dims(np_w_time_2, axis=(1, 2))
    da_hist_norm.data.variances /= numpy.expand_dims(np_w_time_2, axis=(1, 2))
    return da_hist_norm


def normalize_da_hist_by_vanadium_over_angles(da_hist, da_hist_vanadium):
    np_w = da_hist_vanadium.sum("toa").values
    np_w /= np_w.max()
    np_w_unique = numpy.unique(np_w)
    np_w[np_w == 0.0] = np_w_unique[1]
    da_hist_norm = da_hist.copy()
    da_hist_norm.data.values /= np_w
    da_hist_norm.data.variances /= np_w
    return da_hist_norm


def normalize_da_hist_by_vanadium(
    da_hist, da_hist_vanadium, factor_time: float = 0.03
):
    da_hist_norm = normalize_da_hist_by_vanadium_over_angles(
        da_hist, da_hist_vanadium
    )
    da_hist_norm = normalize_da_hist_by_vanadium_over_time(
        da_hist_norm, da_hist_vanadium, factor=factor_time
    )
    return da_hist_norm


def find_peaks_hist(data_event_hist, threshold: float = 0.1):
    # Threshold from 0. to 1.
    """Find peaks by events"""
    np_data = data_event_hist.values
    np_flag_peaks = (
        np_data > threshold * (np_data.max() - np_data.min()) + np_data.min()
    )
    np_flag_peaks = binary_dilation(np_flag_peaks, iterations=2).astype(
        np_flag_peaks.dtype
    )
    peak_labels, peak_number = label(np_flag_peaks)
    print(f"Number of peaks is {peak_number}")
    peak_centers = center_of_mass(
        np_data, peak_labels, range(1, peak_number + 1)
    )

    np_data = data_event_hist.data.values

    print("# 1) centers of mass (fast C implementation)")
    peak_centers = center_of_mass(
        np_data, peak_labels, range(1, peak_number + 1)
    )

    print("# 2) coordinate grids: toa, nu, gamma")
    x, y, z = numpy.indices(peak_labels.shape)
    x_sq, y_sq, z_sq = numpy.square(x), numpy.square(y), numpy.square(z)

    print("# 3) compute weighted sums of squared coordinates")
    sum_w = numpy.array(
        [ndi_sum(np_data, peak_labels, i) for i in range(1, peak_number + 1)]
    )
    sum_x2 = numpy.array(
        [
            ndi_sum(np_data * x**2, peak_labels, i)
            for i in range(1, peak_number + 1)
        ]
    )
    sum_y2 = numpy.array(
        [
            ndi_sum(np_data * y**2, peak_labels, i)
            for i in range(1, peak_number + 1)
        ]
    )
    sum_z2 = numpy.array(
        [
            ndi_sum(np_data * z**2, peak_labels, i)
            for i in range(1, peak_number + 1)
        ]
    )

    print("# 4) second moments")
    mean_z2 = sum_z2 / sum_w
    mean_y2 = sum_y2 / sum_w
    mean_x2 = sum_x2 / sum_w

    print("# 5) first moments (centers)")
    cx, cy, cz = numpy.array(peak_centers).T

    print("# 6) variances")
    var_x = mean_x2 - cx**2
    var_y = mean_y2 - cy**2
    var_z = mean_z2 - cz**2

    np_ind_toa = numpy.array(peak_centers)[:, 0]
    np_ind_nu = numpy.array(peak_centers)[:, 1]
    np_ind_gamma = numpy.array(peak_centers)[:, 2]

    np_hist_toa = data_event_hist.coords["toa"].values
    np_hist_gamma = data_event_hist.coords["event_gamma"].values
    np_hist_nu = data_event_hist.coords["event_nu"].values

    np_toa = numpy.interp(np_ind_toa, range(np_hist_toa.size), np_hist_toa)
    np_gamma = numpy.interp(
        np_ind_gamma, range(np_hist_gamma.size), np_hist_gamma
    )
    np_nu = numpy.interp(np_ind_nu, range(np_hist_nu.size), np_hist_nu)

    sig_toa = (
        numpy.interp(
            np_ind_toa + numpy.sqrt(var_x),
            range(np_hist_toa.size),
            np_hist_toa,
        )
        - np_toa
    )
    sig_gamma = (
        numpy.interp(
            np_ind_gamma + numpy.sqrt(var_y),
            range(np_hist_gamma.size),
            np_hist_gamma,
        )
        - np_gamma
    )
    sig_nu = (
        numpy.interp(
            np_ind_nu + numpy.sqrt(var_z), range(np_hist_nu.size), np_hist_nu
        )
        - np_nu
    )

    return np_toa, np_gamma, np_nu, sig_toa, sig_gamma, sig_nu


def assign_event_peak_to_da(
    data_event,
    sc_toa,
    sc_gamma,
    sc_nu,
    sc_sig_toa,
    sc_sig_gamma,
    sc_sig_nu,
    range_sigma: float = 2.0,
):
    np_event_peak = numpy.zeros((data_event.size,), dtype=int)
    sc_toa = sc_toa.to(unit=data_event.coords["toa"].unit)
    sc_gamma = sc_gamma.to(unit=data_event.coords["event_gamma"].unit)
    sc_nu = sc_nu.to(unit=data_event.coords["event_nu"].unit)
    sc_sig_toa = sc_sig_toa.to(unit=data_event.coords["toa"].unit)
    sc_sig_gamma = sc_sig_gamma.to(unit=data_event.coords["event_gamma"].unit)
    sc_sig_nu = sc_sig_nu.to(unit=data_event.coords["event_nu"].unit)
    for i_peak in range(sc_toa.size):
        print(f"{100*(i_peak+1)/sc_toa.size:.1f}%", end="\r")
        toa, gamma, nu = sc_toa[i_peak], sc_gamma[i_peak], sc_nu[i_peak]
        stoa, sgamma, snu = sc_sig_toa[i_peak], sc_sig_gamma[i_peak], sc_sig_nu[i_peak]
        toa_min, toa_max = toa - range_sigma * stoa, toa + range_sigma * stoa
        gamma_min, gamma_max = (
            gamma - range_sigma * sgamma,
            gamma + range_sigma * sgamma,
        )
        nu_min, nu_max = nu - range_sigma * snu, nu + range_sigma * snu


        flag_toa = sc.logical_and(
            data_event.coords["toa"] >  toa_min,
            data_event.coords["toa"] < toa_max,
        )
        flag_gamma = sc.logical_and(
            data_event.coords["event_gamma"] > gamma_min,
            data_event.coords["event_gamma"] < gamma_max,
        )
        flag_nu = sc.logical_and(
            data_event.coords["event_nu"] > nu_min,
            data_event.coords["event_nu"] < nu_max,
        )
        np_flag = sc.logical_and(
            flag_toa, sc.logical_and(flag_gamma, flag_nu)
        ).values
        np_event_peak[np_flag] = i_peak + 1
    data_event.coords["event_peak"] = sc.array(
        dims=[
            "event",
        ],
        values=np_event_peak,
        dtype=int,
    )
    return


def calc_da_peaks_for_event_peak(da_event: sc.DataArray) -> sc.DataArray:
    l_weight, l_q_aver, l_var_q_aver = [], [], []
    for i_peak in range(da_event.coords["event_peak"].max()):
        flag_peak = da_event.coords["event_peak"] == i_peak + 1
        np_w = numpy.expand_dims(da_event.data[flag_peak].values, axis=1)
        np_q = da_event.coords["Q_vec_rot"][flag_peak].values
        weight = numpy.sum(np_w)
        np_q_aver = (np_q * np_w).sum(axis=0) / weight
        np_var_q_aver = (numpy.square(np_q) * np_w).sum(
            axis=0
        ) / weight - numpy.square(np_q_aver)
        l_weight.append(weight)
        l_q_aver.append(np_q_aver)
        l_var_q_aver.append(np_var_q_aver)

    intensity = sc.array(dims=["peaks"], values=l_weight, unit="counts")
    peaks_q = sc.vectors(
        dims=["peaks"], values=l_q_aver, unit=da_event.coords["Q_vec_rot"].unit
    )
    sigma_peaks_q = sc.vectors(
        dims=["peaks"],
        values=numpy.sqrt(l_var_q_aver),
        unit=da_event.coords["Q_vec_rot"].unit,
    )
    da_peaks = sc.DataArray(
        data=intensity,
        coords={
            "Q_vec_rot": peaks_q,
            "sigma_Q_vec_rot": sigma_peaks_q,
        },
    )
    return da_peaks


def assign_dg_to_da_coords(
    dg: sc.DataGroup, da: sc.DataArray, prefix: str = ""
):
    for s_key in dg.keys():
        da.coords[f"{prefix}_{s_key}"] = dg[s_key]
