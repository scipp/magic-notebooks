
import numpy as np
from skimage import filters, morphology, measure
import scipy.ndimage as ndi


def find_peaks_skimage(np_array_nd,
                       threshold: float = 0.1,
                       max_peak_number: int = 1000,
                       flag_variance: bool = True):

    # --- 1) Adaptive threshold ---
    arr_min = np_array_nd.min()
    arr_max = np_array_nd.max()
    arr_range = arr_max - arr_min

    # Convert your threshold definition to absolute value
    thr = arr_min + threshold * arr_range

    # If threshold is too low → use Otsu as fallback
    if threshold < 0.01:
        thr = filters.threshold_otsu(np_array_nd)

    # Binary mask
    mask = np_array_nd > thr

    # Slight dilation to merge small clusters
    mask = morphology.dilation(mask, morphology.ball(2))

    # Connected components (very fast)
    labels = measure.label(mask, connectivity=1)

    peak_number = labels.max()

    # If too many peaks → increase threshold iteratively
    while peak_number > max_peak_number:
        threshold += 0.05
        thr = arr_min + threshold * arr_range
        mask = np_array_nd > thr
        mask = morphology.dilation(mask, morphology.ball(2))
        labels = measure.label(mask, connectivity=1)
        peak_number = labels.max()

    # --- 2) Region properties ---
    props = measure.regionprops(labels, intensity_image=np_array_nd)

    ndim = np_array_nd.ndim
    peak_centers = np.zeros((ndim, peak_number), dtype=float)

    for i, p in enumerate(props):
        peak_centers[:, i] = p.centroid_weighted  # float indices

    # --- 3) Interpolated peak values ---
    peak_values = ndi.map_coordinates(
        np_array_nd,
        peak_centers,
        order=1,
        mode='nearest'
    )

    if not flag_variance:
        return peak_centers, None, peak_values

    # --- 4) Variances (vectorized) ---
    coords = np.indices(np_array_nd.shape)
    var_xyz = np.zeros((ndim, peak_number), dtype=float)

    # Weighted sum per region
    sum_w = np.array([p.image_intensity.sum() for p in props])

    for ax in range(ndim):
        x = coords[ax]
        sum_x2 = np.array([
            (p.image_intensity * x[p.slice]**2).sum()
            for p in props
        ])
        mean_x2 = sum_x2 / sum_w
        var_xyz[ax] = mean_x2 - peak_centers[ax]**2

    return peak_centers, var_xyz, peak_values