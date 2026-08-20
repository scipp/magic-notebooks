import numpy
import scipp as sc
import magic_graphs


def naive_integration(da, integration_box, scale: float = 1.0):
    ls_out = ["# Naive integration\n"]
    da = da.transform_coords(
        ("two_theta", "h", "k", "l"),
        graph={
            **magic_graphs.graph_detector,
            **magic_graphs.graph_qvec,
            **magic_graphs.graph_hkl,
        },
    )
    if "detector_border" in da.masks.keys():
        flag_not_border = ~da.masks["detector_border"]
        np_hkl = numpy.array(
            [
                da[flag_not_border].coords["h"].values,
                da[flag_not_border].coords["k"].values,
                da[flag_not_border].coords["l"].values,
            ],
            dtype=float,
        )
    else:
        np_hkl = numpy.array(
            [
            da.coords["h"].values,
            da.coords["k"].values,
            da.coords["l"].values,
            ],
            dtype=float,
        )

    np_hkl = numpy.unique(numpy.round(np_hkl, 0).astype(int), axis=1)

    l_fsq_exp = []
    l_fsq_mod = []
    l_hkl = []
    l_ratio = []
    l_wavelength = []
    l_tth = []
    N_hkl = np_hkl.shape[1]
    i_hkl = 0
    for hkl in np_hkl.transpose():
        i_hkl += 1
        print(f"Progress: {100*i_hkl/N_hkl:.2f}", end="\r")
        flag_h = sc.abs(da.coords["h"] - hkl[0]) < integration_box[0]
        flag_k = sc.abs(da.coords["k"] - hkl[1]) < integration_box[1]
        flag_l = sc.abs(da.coords["l"] - hkl[2]) < integration_box[2]
        flag_hkl = sc.logical_and(flag_h, sc.logical_and(flag_k, flag_l))
        da_one_hkl = da[flag_hkl]
        np_wavelength = da_one_hkl.coords["wavelength"].values
        wavelength = numpy.mean(np_wavelength)
        np_tth = da_one_hkl.coords["two_theta"].values
        tth = numpy.mean(np_tth)
        sin_sq = numpy.square(numpy.sin(0.5 * tth))
        val = sc.sum(da_one_hkl.data)
        iint = val.value
        siint = numpy.sqrt(val.variance)
        fsq_exp = scale * iint * sin_sq / (numpy.power(wavelength, 4))
        l_fsq_exp.append(fsq_exp)
        l_hkl.append(hkl)
        l_wavelength.append((wavelength, numpy.std(np_wavelength)))
        l_tth.append((numpy.degrees(tth), numpy.degrees(numpy.std(np_tth))))
    np_fsq_exp = numpy.array(l_fsq_exp)  # *scale_new/scale
    np_wavelength = numpy.array(l_wavelength, dtype=float)

    np_tth = numpy.array(l_tth, dtype=float)
    np_hkl_int = numpy.array(l_hkl, dtype=int).transpose()
    return np_hkl_int, np_fsq_exp, np_wavelength, np_tth



import numpy as np

def integrate_peaks_md_box_memorysafe(
    events_xyz,
    events_weight,
    peaks_xyz,
    box_half_sizes,
    bg_inner_half_sizes=None,
    bg_outer_half_sizes=None,
    metadata_mask=None,
    preassign_nonoverlapping=False,
):
    """
    Memory-safe Mantid-independent box integration.
    - No KD-tree
    - No dimension expansion (never allocates NxM arrays)
    - Peak-by-peak processing
    - Optional metadata filtering
    - Optional non-overlapping peak preassignment (streaming-safe)

    Parameters
    ----------
    events_xyz : (N,3) float array
        Event coordinates.
    events_weight : (N,) float array
        Event weights.
    peaks_xyz : (M,3) float array
        Peak centers.
    box_half_sizes : (3,) float array
        Half-size of integration box.
    bg_inner_half_sizes : (3,) float array or None
        Inner background box.
    bg_outer_half_sizes : (3,) float array or None
        Outer background box.
    metadata_mask : (N,) bool array or None
        Pre-filter mask for events.
    preassign_nonoverlapping : bool
        If True, assign events to peaks once (streaming-safe).

    Returns
    -------
    intensities : (M,) float array
    errors : (M,) float array
    """

    # --- Step 1: metadata filtering ---
    if metadata_mask is not None:
        events_xyz = events_xyz[metadata_mask]
        events_weight = events_weight[metadata_mask]

    N = len(events_xyz)
    M = len(peaks_xyz)

    intensities = np.zeros(M)
    errors = np.zeros(M)

    # --- Step 2: optional preassignment (memory-safe) ---
    # We assign each event to at most one peak, but we do it WITHOUT NxM arrays.
    if preassign_nonoverlapping:
        event_to_peak = np.full(N, -1, dtype=np.int64)

        for j, peak in enumerate(peaks_xyz):
            diff = np.abs(events_xyz - peak)
            mask_peak = np.all(diff <= box_half_sizes, axis=1)

            # Assign only events not yet assigned
            unassigned = (event_to_peak == -1)
            event_to_peak[mask_peak & unassigned] = j

    # --- Step 3: integrate peak-by-peak ---
    for j, peak in enumerate(peaks_xyz):

        if preassign_nonoverlapping:
            mask_peak = (event_to_peak == j)
        else:
            diff = np.abs(events_xyz - peak)
            mask_peak = np.all(diff <= box_half_sizes, axis=1)

        w_peak = events_weight[mask_peak]
        I_peak = np.sum(w_peak)
        E_peak = np.sqrt(np.sum(w_peak**2))

        # --- Background ---
        if bg_inner_half_sizes is not None and bg_outer_half_sizes is not None:

            diff = np.abs(events_xyz - peak)

            mask_outer = np.all(diff <= bg_outer_half_sizes, axis=1)
            mask_inner = np.all(diff <= bg_inner_half_sizes, axis=1)
            mask_bg = mask_outer & (~mask_inner)

            w_bg = events_weight[mask_bg]
            I_bg = np.sum(w_bg)
            E_bg = np.sqrt(np.sum(w_bg**2))

            # Volume scaling
            V_peak = np.prod(2 * box_half_sizes)
            V_bg = np.prod(2 * bg_outer_half_sizes) - np.prod(2 * bg_inner_half_sizes)

            bg_scaled = I_bg * (V_peak / V_bg)
            bg_err_scaled = E_bg * (V_peak / V_bg)

            intensities[j] = I_peak - bg_scaled
            errors[j] = np.sqrt(E_peak**2 + bg_err_scaled**2)

        else:
            intensities[j] = I_peak
            errors[j] = E_peak

    return intensities, errors


def boxes_overlap(center1, half1, center2, half2):
    """
    Check if two axis-aligned boxes overlap.
    Overlap occurs if intervals intersect along ALL axes.
    """
    return np.all(np.abs(center1 - center2) <= (half1 + half2))


def check_peak_background_overlaps(
    peaks_xyz,
    peak_half_sizes,
    bg_inner_half_sizes,
    bg_outer_half_sizes,
):
    """
    Check for forbidden overlaps between peak and background boxes.

    Forbidden:
      1. Peak box overlaps with another peak box
      2. Peak box overlaps with background box of another peak

    Allowed:
      - Background boxes may overlap with each other
      - Background boxes may overlap with outer background boxes of other peaks

    Parameters
    ----------
    peaks_xyz : (M,3) array
        Peak centers.
    peak_half_sizes : (3,) array
        Half-size of peak integration box.
    bg_inner_half_sizes : (3,) array
        Inner background box half-size.
    bg_outer_half_sizes : (3,) array
        Outer background box half-size.

    Returns
    -------
    ok : bool
        True if no forbidden overlaps exist.
    problems : list of str
        Human-readable descriptions of detected overlaps.
    """

    M = len(peaks_xyz)
    problems = []
    l_ind = []
    for i in range(M):
        for j in range(i + 1, M):

            pi = peaks_xyz[i]
            pj = peaks_xyz[j]

            # --- 1. Peak vs Peak overlap (forbidden) ---
            if boxes_overlap(pi, peak_half_sizes, pj, peak_half_sizes):
                problems.append(
                    f"Peak {i} overlaps with peak {j} (peak region overlap)."
                )
                l_ind.extend([i, j])

            # --- 2. Peak i vs Background j (forbidden) ---
            if boxes_overlap(pi, peak_half_sizes, pj, bg_outer_half_sizes):
                problems.append(
                    f"Peak {i} overlaps with background of peak {j}."
                )
                l_ind.extend([i, j])

            # --- 3. Peak j vs Background i (forbidden) ---
            if boxes_overlap(pj, peak_half_sizes, pi, bg_outer_half_sizes):
                problems.append(
                    f"Peak {j} overlaps with background of peak {i}."
                )
                l_ind.extend([i, j])

    ok = (len(problems) == 0)
    return ok, problems, sorted(set(l_ind))
