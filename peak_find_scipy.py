import numpy as np
import scipy.ndimage as ndi

def find_peaks_in_np_array_nd(np_array_nd,
                              threshold: float = 0.1,
                              max_peak_number: int = 1000,
                              flag_variance: bool = True):

    blurred = ndi.gaussian_filter(np_array_nd, sigma=20)
    np_array_nd_diff = np_array_nd - blurred
    np_array_nd_diff[np_array_nd_diff<0.] = 0.
    # --- 1) Adaptive thresholding ---
    if threshold < 1e-5:
        threshold = 1e-5
    peak_number = np.inf
    delimeter = abs(1 / threshold)
    if delimeter <= 1.:
        delimeter = 10.
    threshold = 1. / delimeter

    arr_min = np_array_nd_diff.min()
    arr_max = np_array_nd_diff.max()
    arr_range = arr_max - arr_min

    while peak_number > max_peak_number:
        np_flag_peaks = np_array_nd_diff > (threshold * arr_range + arr_min)
        np_flag_peaks = ndi.binary_dilation(np_flag_peaks, iterations=2)

        peak_labels, peak_number = ndi.label(np_flag_peaks)

        if peak_number > max_peak_number:
            delimeter += 1.
            threshold += 1. / delimeter

    # --- 2) Centers of mass ---
    peak_centers = np.array(
        ndi.center_of_mass(np_array_nd, peak_labels, range(1, peak_number + 1))
    ).T   # shape (ndim, n_peaks)

    # --- 3) Interpolated peak values (NEW) ---
    # Use linear interpolation via map_coordinates
    peak_values = ndi.map_coordinates(
        np_array_nd,
        peak_centers,
        order=1,  # linear interpolation
        mode='nearest'
    )

    if not flag_variance:
        return peak_centers, None, peak_values

    # --- 4) Weighted sums (vectorized) ---
    sum_w = ndi.sum(np_array_nd, peak_labels, index=range(1, peak_number + 1))

    # --- 5) Variances (vectorized) ---
    ndim = np_array_nd.ndim
    var_xyz = np.empty((ndim, peak_number), dtype=float)

    # Precompute coordinate grids once
    coords = np.indices(np_array_nd.shape)

    for ax in range(ndim):
        x = coords[ax]
        sum_x2 = ndi.sum(np_array_nd * x * x, peak_labels, index=range(1, peak_number + 1))
        mean_x2 = sum_x2 / sum_w
        var_xyz[ax] = mean_x2 - peak_centers[ax] ** 2

    return peak_centers, var_xyz, peak_values

