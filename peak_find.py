from scipy.spatial import cKDTree
import numpy
import scipp as sc

def adaptive_bandwidth(q):
    # Replace with your MAGiC/ESS resolution model
    Q = numpy.linalg.norm(q)
    return 0.01 + 0.02 * Q


class MeanShiftNeighborSearch:
    def __init__(self, q_events):
        self.q_events = numpy.asarray(q_events)
        self.tree = cKDTree(self.q_events)

    def neighbors_within(self, x, radius):
        idx = self.tree.query_ball_point(x, radius)
        return numpy.asarray(idx, dtype=int)


def mean_shift_single_accel(
    q_events,
    weights,
    start,
    max_iter=200,
    tol=1e-6,
    radius_factor=3.0,
    neighbor_search=None,
):
    """
    One adaptive mean-shift run using KD-tree neighbor search.
    """
    if neighbor_search is None:
        neighbor_search = MeanShiftNeighborSearch(q_events)

    x = start.copy()
    for _ in range(max_iter):
        h = adaptive_bandwidth(x)
        radius = radius_factor * h

        idx = neighbor_search.neighbors_within(x, radius)
        if idx.size == 0:
            break

        q_loc = q_events[idx]
        w_loc = weights[idx]

        diff = q_loc - x
        dist2 = numpy.sum(diff**2, axis=1)

        k = numpy.exp(-dist2 / (2 * h * h))
        w = w_loc * k
        denom = numpy.sum(w)
        if denom == 0:
            break

        x_new = numpy.sum(q_loc * w[:, None], axis=0) / denom
        if numpy.linalg.norm(x_new - x) < tol:
            return x_new

        x = x_new

    return x
def _cluster_modes(modes, merge_radius):
    clusters = []
    for m in modes:
        placed = False
        for c in clusters:
            if numpy.linalg.norm(m - c) < merge_radius:
                c[:] = 0.5 * (c + m)
                placed = True
                break
        if not placed:
            clusters.append(m.copy())
    return numpy.array(clusters)


def _compute_peak_intensity(q_events, weights, peak, radius, neighbor_search):
    idx = neighbor_search.neighbors_within(peak, radius)
    if idx.size == 0:
        return 0.0
    return numpy.sum(weights[idx])


def find_multiple_peaks_accel(
    events_coords,
    events_weight=None,
    # max_peaks=10,
    merge_radius=0.05,
    basin_radius=0.1,
    max_seeds=5000,
    random_state=None,
    radius_factor=3.0,
):
    """
    Multi-peak adaptive mean-shift with KD-tree acceleration.
    """
    q_events = events_coords.values


    q_events = numpy.asarray(q_events)
    N, d = q_events.shape

    if events_weight is None:
        weights = numpy.ones(N, dtype=float)
    else:
        weights = numpy.asarray(events_weight.values, dtype=float)


    rng = numpy.random.default_rng(random_state)
    if N > max_seeds:
        idx_seeds = rng.choice(N, size=max_seeds, replace=False)
    else:
        idx_seeds = numpy.arange(N)

    neighbor_search = MeanShiftNeighborSearch(q_events)

    modes = []
    for i in idx_seeds:
        m = mean_shift_single_accel(
            q_events,
            weights,
            start=q_events[i],
            neighbor_search=neighbor_search,
            radius_factor=radius_factor,
        )
        modes.append(m)
    modes = numpy.array(modes)

    unique_peaks = _cluster_modes(modes, merge_radius)

    intensities = numpy.array([
        _compute_peak_intensity(
            q_events, weights, p, basin_radius, neighbor_search
        )
        for p in unique_peaks
    ])

    idx = numpy.argsort(intensities)[::-1]#[:max_peaks]
    peaks_q = sc.vectors(dims=['peaks'], values=unique_peaks[idx], unit=events_coords.unit)
    if events_weight is None:
        peaks_intensity =sc.array(dims=['peaks'], values=intensities[idx], unit='counts')
    else:
        peaks_intensity =sc.array(dims=['peaks'], values=intensities[idx], unit=events_weight.unit)


    da = sc.DataArray(
                data=sc.array(
                    dims=['peaks'], values=intensities[idx]
                    ),
            coords={
                'Q_vec_rot': peaks_q,
            }
            )
    return da