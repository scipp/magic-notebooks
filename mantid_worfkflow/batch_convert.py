#!/usr/bin/env python3
"""
batch_convert.py - McStas MAGiC runs -> CODA-format NeXus files, one or many.

THE converter of this project.  mcstas_to_nexus.py, _v2 and _v3 did the
single-file part and are archived (archive/converters/); their code lives here
now, so there is one script and no second copy to drift apart.

    python3 batch_convert.py --mcstas 'mysample_*.h5' \
        --template coda_magic_999999_00016485.hdf --tof-window 22,92

    --template is the CODA file whose NeXus layout the output copies; it is
    always given, because the same chain is run against different CODA files
    (section 122).  The helper modules are looked for in dependencies/ next to
    this script (or next to it directly).  The instrument definition is
    built from voxelization.py - the model that assigns the detector IDs - for
    every batch (section 120), written to MAGiC_Definition_vox.xml in --outdir
    and embedded in every file; --idf FILE embeds a given one instead.

    --tof-window is in TRUE flight time.  A window that reaches beyond the
    71.43 ms frame (e.g. 22,92: lambda 0.54..2.26 A, as an ideal bandwidth
    chopper one frame wide) writes the slow neutrons into the next pulse, as
    the detector records them; magic_workflow.py --unwrap auto undoes it.

    --dry-run reads and prints every angle and converts nothing.

What goes into each file
------------------------
* events of bank A (and B, if McStas has them) and of the cave monitor, drawn
  from the McStas weights, spread over --pulses pulses at 14 Hz;
* only events inside the real (trapezoidal) segments: read_h5.py cuts every
  McStas event against voxelization.py, the reference geometry, and prints how
  many survive (plan sections 96-97);
* with --geometry, every event on the voxel nearest to where McStas recorded
  it.  The mesh in mccode_v2_geometry.npz comes from the CODA detector_faces,
  which is OUTDATED (Iurii, 28 Sep 2026); the number of events further from
  their voxel centre than half a voxel diagonal is printed as the check that
  must go to ~0 once the new mesh is in;
* the goniometer angle, in entry/instrument/sample_stick_rotation/value, the
  NXlog where a CODA file carries it.  McStas stores omega as an expression,
  so it is taken from the sample's rotation matrix: log = sense*angle + offset,
  defaults -1 and -90, which reproduce the orientations measured
  independently from the Fe4O5 runs (plan section 87).  Mantid does not read
  logs from an NXpositioner; magic_workflow.py reads it with h5py;
* detector_a_rotation / detector_b_rotation, the bank angles;
* L1 from McStas (the template's source position is not this instrument's);
* each run its own run number and its own start time, the pulses moved with it,
  spaced by the run duration plus one second;
* the instrument definition, built from voxelization.py and embedded (--idf
  FILE embeds a given one; --build-idf the old mesh-based one).

What does NOT go in: t0.  No MAGiC_Parameters.xml is written (the old default
wrote t0 = 3000 us there, and Mantid picking it up or not is what broke the
t0 bookkeeping - plan section 85).  t0 belongs to the reduction.
"""
import argparse
import glob
import os
import shutil
import sys
import traceback

import h5py as h5
import numpy as np
import scipp as sc

# section 120: helper modules and the CODA template live in dependencies/
HERE = os.path.dirname(os.path.abspath(__file__))
DEPENDENCIES = os.path.join(HERE, "dependencies")
if os.path.isdir(DEPENDENCIES):
    sys.path.insert(0, DEPENDENCIES)
import read_h5                                                 # noqa: E402


PULSE_FREQUENCY_HZ = 14.0


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def sample_event_indices(event_probability, number_event, rng=None, poisson=False):
    """Pick `number_event` event indices distributed according to the weights.

    CHANGED 4: the input array is never modified, the sampler is seeded, and
    the default path is an inverse-CDF draw, which is O(N + n log N) instead of
    the O(N log N) of np.random.choice and does not build an extra probability
    table for 1e8 events.
    """
    rng = np.random.default_rng() if rng is None else rng
    weights = np.asarray(event_probability, dtype=float)
    if np.any(weights < 0):
        raise ValueError("negative McStas weights")
    total = weights.sum()
    if total <= 0:
        raise ValueError("all McStas weights are zero")

    if poisson:
        # each simulated neutron is used at most as often as it deserves; no
        # duplicate-free guarantee, but the events stay statistically independent
        counts = rng.poisson(weights * (number_event / total))
        return np.repeat(np.arange(weights.size), counts)

    cdf = np.cumsum(weights)
    draws = rng.random(number_event) * cdf[-1]
    return np.searchsorted(cdf, draws, side="right")


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------
def replace_dataset(entry, name, values, dtype=None):
    """Replace a dataset, keeping attributes AND the original dtype.

    CHANGED 1: the old version took the dtype of whatever numpy array was
    passed in, so int32/uint64 fields silently became float64.
    """
    old = entry[name]
    attrs = dict(old.attrs)
    target = old.dtype if dtype is None else np.dtype(dtype)
    values = np.asarray(values)
    if target.kind in "iu" and values.dtype.kind == "f":
        rounded = np.rint(values)
        if values.size and np.abs(values - rounded).max() > 1.e-6:
            print("      note: rounding %s to %s" % (name, target))
        values = rounded
    del entry[name]
    dset = entry.create_dataset(name, data=values.astype(target, copy=False))
    dset.attrs.update(attrs)
    return dset


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.dtype.kind in "SUO" and value.size == 1:
        return decode(value.flat[0])
    return value


def set_string(group, name, text):
    """Write a fixed length string, replacing whatever was there."""
    attrs = dict(group[name].attrs) if name in group else {}
    if name in group:
        del group[name]
    dset = group.create_dataset(name, data=np.bytes_(str(text).encode("utf-8")))
    dset.attrs.update(attrs)
    return dset


def template_start_time(path):
    """Run start of the template as (nanoseconds since epoch, ISO string).

    Plain integers on purpose: sc.datetime() takes its unit from the number of
    digits in the string ("...48.433" is milliseconds), which is what produced
    "Cannot add ns and ms".
    """
    text = "2026-01-01T12:00:00"
    with h5.File(path, "r") as f:
        if "entry/start_time" in f:
            text = str(decode(f["entry/start_time"][()]))
    stamp = np.datetime64(text.replace("Z", "").split("+")[0], "ns")
    return int(stamp.astype("int64")), str(stamp)


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------
def apply_tof_window(toa_ns, event_id, weights, window_ms):
    """CHANGED 14: keep only the events inside a time-of-flight window.

    The simulation has no bandwidth chopper: at L = 160.7 m its 0.52..2.37 A band
    is 21..96 ms of flight time, but the frame is 71.4 ms. Everything slower than
    1.76 A therefore wraps into the next frame, comes back as an apparently fast
    neutron below 0.61 A, and dilutes the Bragg peaks into background. The real
    instrument cuts the band with choppers; --tof-window does the same here.
    """
    if not window_ms:
        return toa_ns, event_id, weights
    lo, hi = [float(v) * 1.0e6 for v in window_ms]
    keep = (toa_ns >= lo) & (toa_ns <= hi)
    print("      TOF window %.1f..%.1f ms: %d of %d events kept (%.1f %%)"
          % (lo * 1e-6, hi * 1e-6, int(keep.sum()), keep.size,
             100.0 * keep.mean()))
    # Section 98: the window is in TRUE flight time, as an ideal bandwidth
    # chopper would pass it; build_event_arrays() then folds every time into
    # the 71.43 ms frame, so an event of the window beyond the frame edge is
    # written into the NEXT pulse at t - 71.43 ms, as the detector records it.
    # magic_workflow.py --unwrap takes it back.  A window longer than one
    # frame makes the two pulses overlap in the recorded frame.
    period = 1.0e9 / PULSE_FREQUENCY_HZ
    folded = keep & (toa_ns >= period)
    if folded.any():
        print("      %d kept events (%.1f %%) lie beyond the %.2f ms frame edge"
              " and are written into the next pulse at t - %.2f ms (previous-"
              "pulse neutrons; magic_workflow.py --unwrap)"
              % (int(folded.sum()), 100.0 * folded.sum() / max(keep.sum(), 1),
                 period * 1e-6, period * 1e-6))
        print("      recorded frame: %.2f..%.2f ms from the previous pulse, "
              "empty %.2f..%.2f ms, %.2f..%.2f ms from this pulse"
              % (0.0, (hi - period) * 1e-6, (hi - period) * 1e-6, lo * 1e-6,
                 lo * 1e-6, period * 1e-6))
    if hi - lo >= period:
        print("      WARNING: the window is %.2f ms long, the frame %.2f ms: "
              "the two pulses overlap in the recorded frame and no split "
              "can separate them" % ((hi - lo) * 1e-6, period * 1e-6))
    if not np.any(keep):
        raise ValueError("the TOF window keeps no events at all")
    return toa_ns[keep], np.asarray(event_id)[keep], \
        (None if weights is None else np.asarray(weights)[keep])


def build_event_arrays(toa_ns, event_id, start_ns, n_pulses=1, rng=None,
                       frequency=PULSE_FREQUENCY_HZ):
    """Turn absolute arrival times into the four CODA event datasets.

    CHANGED 10: the events can be spread over `n_pulses` pulses instead of
    landing in the one or two frames the McStas times happen to cover.

    toa_ns    arrival time of each event, nanoseconds (McStas time origin)
    event_id  detector ID of each event
    start_ns  time of pulse 0, nanoseconds since the epoch
    n_pulses  how many pulses the run is spread over (1 = old behaviour)

    Each event keeps its time of flight: the frame it belongs to is
    floor(toa / period), and a random pulse in [0, n_pulses) is added on top, so
    slow neutrons still arrive in a later frame than they were emitted in.
    Every pulse of the run appears in event_time_zero, including the empty ones,
    exactly as in a real CODA file.
    """
    rng = np.random.default_rng() if rng is None else rng
    period_ns = 1.0e9 / float(frequency)

    toa_ns = np.asarray(toa_ns, dtype=float)
    event_id = np.asarray(event_id)
    frame = np.floor(toa_ns / period_ns).astype(np.int64)
    offset = toa_ns - frame * period_ns                 # time of flight in the frame

    n_pulses = max(1, int(n_pulses))
    if n_pulses > 1:
        pulse = frame + rng.integers(0, n_pulses, size=toa_ns.size)
    else:
        pulse = frame

    order = np.lexsort((offset, pulse))                 # NeXus wants pulse-major order
    pulse, offset, event_id = pulse[order], offset[order], event_id[order]

    total_pulses = int(pulse.max()) + 1 if pulse.size else n_pulses
    total_pulses = max(total_pulses, n_pulses)
    counts = np.bincount(pulse, minlength=total_pulses)
    event_index = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    event_time_zero = start_ns + np.rint(
        np.arange(total_pulses) * period_ns).astype(np.int64)

    return {"event_id": event_id,
            "event_time_offset": offset,
            "event_time_zero": event_time_zero,
            "event_index": event_index,
            "n_pulses": total_pulses,
            "duration_s": total_pulses / float(frequency)}


def apply_delta_t(toa_ns, delta_t_ms):
    """Shift the arrival times by the long-pulse correction delta_t (ms).

    delta_t belongs to the time origin (the emission time inside the 3 ms ESS
    pulse), so shifting toa is the same as shifting every time of flight.
    OFF BY DEFAULT: delta_t and delta_L are refinable, so the file keeps the raw
    times and the correction lives in MAGiC_Parameters.xml (--write-parameters).
    """
    if not delta_t_ms:
        return toa_ns
    return toa_ns - float(delta_t_ms) * 1.0e6


def ids_from_position(detector_data, event_ind, geometry, label, gamma):
    """CHANGED 15: give every event the voxel it really hit, by position.

    Why this exists. The detector_number McStas writes comes from the analytic
    model in voxelization.py, while MAGiC_Definition.xml is built from the
    detector_faces mesh of the CODA template, and the two describe segments with
    different parameters:

      * vertical binning. The analytic model divides a segment evenly in HEIGHT
        (constant step 7.686 mm, polar angle step varying by 38 %); the mesh
        divides it close to evenly in SCATTERING ANGLE (polar step varying by
        11 %, height step by 38 %). They agree at both ends of the segment and
        part company in the middle, by up to 39 mm at anode 88.
      * segment tilt and length. The mesh segment is 0.5368 m long and tilted
        17.65 deg from the radial direction; voxelization.py has r_vs = 0.530 m
        and omega_vs = -9.101 deg. The azimuths agree at the front face
        (0.03 deg) and drift to 1.31 deg at full depth; fitting omega_vs to the
        mesh would give about -5.7 deg.

    Together: a median transverse error of 1.54 deg. A powder only sees that as
    a per-cent shift in d - which is what the Ge lines showed, and why they
    could be made to agree. A single crystal sees a rotation of Q: 0.10 to
    0.27 A^-1 depending on wavelength, against the 0.060 A^-1 that IndexPeaks
    allows for Fe4O5. No orientation matrix survives it, and
    FindUBUsingLatticeParameters answers with a distorted cell rather than an
    honest failure.

    What is NOT wrong, though it looks it at first: `calc_xyz_by_id` returns one
    position for each PAIR of voxels, and a round trip id -> position -> id
    cannot return the original id for more than half of them. That is the
    physics. Neutrons are captured on a boron layer with a voxel on either side,
    so both voxels of a pair report the same place - the boron plane - and a
    position can never say which of the two fired. Checked against the mesh, the
    analytic planes sit on the midpoints of the mesh pairs to 0.033 deg with no
    scatter.

    Assigning by position takes the analytic model, and with it the id map, out
    of the chain: events are placed on the same mesh Mantid reads, so the two
    agree by construction. Which description is the real detector A stays an
    open question for the instrument team, but it no longer blocks the
    reduction.

    One residual is left on purpose. The event was captured on a boron plane,
    and Mantid will place it at the centre of whichever voxel it is assigned to,
    0.240 deg away - the same either way, since the plane is equidistant from
    both. That is 0.015 A^-1 at 1.75 A and 0.042 A^-1 at 0.62 A, i.e. 0.25 to
    0.71 of the indexing budget: tolerable. An IDF that placed both voxels of a
    pair on their shared boron plane would remove it, and would describe the
    detector as it actually works.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise SystemExit("--assign-by-position needs scipy (conda install scipy)")
    key = "magic_detector_%s" % label
    if key + "_pos" not in geometry:
        raise SystemExit("%s not in the geometry file" % key)
    centres = np.asarray(geometry[key + "_pos"], dtype=float)
    voxel_ids = np.asarray(geometry[key + "_ids"], dtype=np.int64)
    local = detector_data.bins.coords["event_position_local_mcstas"] \
        .bins.concat().value[event_ind].values
    cos, sin = np.cos(gamma), np.sin(gamma)
    rotation = np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
    lab = np.asarray(local, dtype=float) @ rotation.T
    distance, index = cKDTree(centres).query(lab, k=1, workers=-1)
    print("      assigned by position: %d events, distance to the voxel centre "
          "median %.4f m, 90 %% below %.4f m"
          % (lab.shape[0], float(np.median(distance)),
             float(np.quantile(distance, 0.9))))
    print("      distinct voxels hit: %d" % np.unique(index).size)
    # Every event here is inside a voxelization.py segment (read_h5.py cut the
    # rest).  If the mesh described the same detector, each would lie inside
    # its nearest voxel, i.e. within half a voxel diagonal of its centre.
    size = geometry.get(key + "_size") if hasattr(geometry, "get") else None
    if size is None and key + "_size" in geometry:
        size = geometry[key + "_size"]
    if size is not None:
        reach = 0.5*float(np.linalg.norm(np.asarray(size, dtype=float)))
        far = distance > reach
        print("      mesh vs voxelization.py: %d events (%.3f %%) lie further "
              "than half a voxel diagonal (%.4f m) from their mesh voxel -"
              " %s" % (int(far.sum()), 100.0*far.mean(), reach,
                       "the mesh and voxelization.py agree" if far.mean() < 1e-3
                       else "the mesh does not describe the detector of "
                       "voxelization.py (CODA detector_faces is outdated)"))
    # If McStas had already snapped its positions to the analytic voxel grid,
    # assigning by position would buy nothing - say so rather than assume.
    distinct = np.unique(np.round(lab, 6), axis=0).shape[0]
    print("      distinct event positions: %d of %d events (%s)"
          % (distinct, lab.shape[0],
             "continuous, as required" if distinct > 0.5 * lab.shape[0] else
             "SNAPPED to a grid - the McStas positions are quantised, so this "
             "mode cannot help"))
    return voxel_ids[index]


def detector_to_coda_format(da_detector, det_numbers, number_event,
                            start_ns, rng=None, poisson=False, delta_t_ms=0.0,
                            n_pulses=1, tof_window=None,
                            position_geometry=None, label=None):
    """McStas detector events -> arrays in CODA layout."""
    hh = sc.array(values=det_numbers, dims=("detector_number",))
    before = int(da_detector.bins.size().sum().value)
    detector_data = da_detector.group(hh)
    after = int(detector_data.bins.size().sum().value)
    # CHANGED 6: grouping silently drops events whose detector_number is not in
    # the template; say so instead of losing them quietly.
    print("      events in the McStas data : %d" % before)
    print("      events matching detector_number: %d (%.2f %% lost)"
          % (after, 100.0 * (before - after) / max(before, 1)))
    if after == 0:
        raise ValueError("no McStas event matches the detector_number list of the "
                         "template - is the ID numbering 0-based vs 1-based?")

    event_probability = detector_data.bins.data.bins.concat().value.values
    event_ind = sample_event_indices(event_probability, number_event, rng, poisson)

    toa_ns = detector_data.bins.coords["toa"].bins.concat().value[event_ind] \
        .to(unit="ns").values
    if position_geometry is not None:
        ids = ids_from_position(detector_data, event_ind, position_geometry,
                                label, float(da_detector.coords["gamma"].value))
    else:
        ids = sc.bins_like(detector_data,
                           sc.array(dims=detector_data.dims,
                                    values=det_numbers)
                           ).bins.concat().value[event_ind].values

    toa_ns, ids, _ = apply_tof_window(apply_delta_t(toa_ns, delta_t_ms), ids,
                                      None, tof_window)
    result = build_event_arrays(toa_ns, ids, start_ns, n_pulses, rng)
    result["detector_rotation_value"] = detector_data.coords["gamma"]
    return result


def cave_monitor_to_coda_format(da_cave_monitor, number_event, start_ns,
                                rng=None, poisson=False, monitor_id=0,
                                delta_t_ms=0.0, n_pulses=1, tof_window=None):
    """The cave monitor is a TOF histogram, not an event list: draw events from it.

    CHANGED: the sampled times are spread uniformly inside their bin instead of
    all sitting exactly on the bin centre, and event_id is an integer array.
    """
    rng = np.random.default_rng() if rng is None else rng
    event_ind = sample_event_indices(da_cave_monitor.values, number_event,
                                     rng, poisson)

    toa_all = da_cave_monitor.coords["toa"].to(unit="ns").values
    toa_ns = np.asarray(toa_all, dtype=float)[event_ind]
    edges = np.sort(np.unique(toa_all))
    if edges.size > 1:                       # spread inside the histogram bin
        width = float(np.median(np.diff(edges)))
        toa_ns = toa_ns + (rng.random(toa_ns.size) - 0.5) * width

    ids = np.full(toa_ns.size, int(monitor_id), dtype=np.int32)
    toa_ns, ids, _ = apply_tof_window(apply_delta_t(toa_ns, delta_t_ms), ids,
                                      None, tof_window)
    return build_event_arrays(toa_ns, ids, start_ns, n_pulses, rng)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def load_id_map(path):
    """CHANGED 13: table that turns a McStas detector_number into the CODA one.

    build_id_map.py derives it from the two geometry descriptions: for bank A the
    angular index runs the other way (column = 2*N_vs - 1 - n_vs), bank B is the
    identity. Without it the events of bank A land on voxels half a metre away and
    the powder pattern flattens into featureless background.
    """
    if not path:
        return {}
    data = np.load(path)
    maps = {}
    for label in ("magic_detector_a", "magic_detector_b"):
        if label in data:
            maps[label[-1]] = (np.asarray(data[label], dtype=np.int64),
                               int(data[label + "_id0"][0]))
            print("  id map for detector_%s: %d entries, id0 = %d"
                  % (label[-1], maps[label[-1]][0].size, maps[label[-1]][1]))
    return maps


def apply_id_map(event_id, table, id0):
    ids = np.asarray(event_id, dtype=np.int64) - id0
    bad = (ids < 0) | (ids >= table.size)
    if np.any(bad):
        print("      WARNING: %d event(s) outside the id map - left unchanged"
              % int(bad.sum()))
        ids = np.clip(ids, 0, table.size - 1)
    mapped = table[ids]
    mapped[bad] = np.asarray(event_id)[bad]
    changed = np.mean(mapped != np.asarray(event_id))
    print("      id map applied: %.1f %% of the events changed voxel"
          % (100.0 * changed))
    return mapped


def get_det_numbers(f_nexus, label_detector="a"):
    with h5.File(f_nexus, "r") as f:
        return f["entry/instrument/magic_detector_%s/detector_number" % label_detector][()]


def _write_event_group(det_event_data, data):
    replace_dataset(det_event_data, "event_id", data["event_id"])
    replace_dataset(det_event_data, "event_time_offset", data["event_time_offset"])
    dset = replace_dataset(det_event_data, "event_time_zero", data["event_time_zero"])
    # CHANGED 2: Mantid reads 'offset', the CODA files only carry 'start'
    if "offset" not in dset.attrs:
        dset.attrs["offset"] = np.bytes_(
            str(decode(dset.attrs.get("start", b"1970-01-01T00:00:00Z"))).encode())
    replace_dataset(det_event_data, "event_index", data["event_index"])
    filled = int(np.count_nonzero(np.diff(np.append(
        data["event_index"], data["event_id"].size))))
    print("      wrote %d events in %d pulse(s), %d of them non-empty (%.3f s run)"
          % (data["event_id"].size, data["n_pulses"], filled, data["duration_s"]))


def replace_detector_event(f_nexus, data_detector, label_detector="a"):
    with h5.File(f_nexus, "r+") as f:
        det_group = f["entry/instrument/magic_detector_%s" % label_detector]
        _write_event_group(
            det_group["magic_detector_%s_event_data" % label_detector], data_detector)

        rotation = f["entry/instrument/detector_%s_rotation" % label_detector]["value"]
        angle = data_detector["detector_rotation_value"].to(unit="deg").value
        replace_dataset(rotation, "value", np.array([angle], dtype=float))
        replace_dataset(rotation, "time", np.array([0]))
        print("      detector_%s_rotation = %.4f deg" % (label_detector, angle))


def clear_event_group(f_nexus, group_path):
    """Empty an NXevent_data group.

    CHANGED 9: if a detector is not present in the McStas file, the copy of the
    template still holds the template's real events (23 M of them for bank b).
    Leaving them there silently mixes measured data into a simulated file, so
    the group is emptied instead.
    """
    with h5.File(f_nexus, "r+") as f:
        if group_path not in f:
            return False
        group = f[group_path]
        before = group["event_id"].shape[0] if "event_id" in group else 0
        for name in ("event_id", "event_time_offset"):
            if name in group:
                replace_dataset(group, name, np.zeros(0))
        for name in ("event_time_zero", "event_index"):
            if name in group:
                replace_dataset(group, name, np.zeros(0))
        print("      emptied %s (%d template events dropped)" % (group_path, before))
    return True


def replace_monitor_event(f_nexus, data_monitor):
    with h5.File(f_nexus, "r+") as f:
        _write_event_group(f["entry/instrument/beam_monitor_1/beam_monitor_1_events"],
                           data_monitor)


OMEGA_COMPONENTS = ("sampleMantid", "arm_sample_rot_1")


def omega_from_mcstas(mcstas_file, sense=-1.0, offset=-90.0):
    """The goniometer angle of this run, from the sample's rotation matrix.

    McStas stores component parameters as the expressions that produced them,
    so the omega of a run cannot be read as a number - but every component's
    Rotation matrix IS stored as numbers.  For a rotation about the vertical,
    R = [[c, 0, s], [0, 1, 0], [-s, 0, c]] and the McStas angle is
    atan2(R02, R00).  The matrix is rebuilt from that angle and the residual
    reported, so a chi or phi rotation cannot pass silently.

    The log value is  sense*angle + offset.  sense = -1 and offset = -90 are
    measured, not assumed: with them the orientations found independently from
    the three Fe4O5 runs are related by exactly the logged differences
    (+90.0 and +126.9 deg about +y, i.e. -90 and -53 modulo the 2-fold about b),
    and run 1 sits at 0 (plan sections 17.1 and 87).  Mantid then takes it with
    SetGoniometer(Axis0="sample_stick_rotation,0,1,0,1").
    Returns (log value, McStas angle, residual, component) or None.
    """
    try:
        with h5.File(mcstas_file, "r") as handle:
            components = handle["entry1/instrument/components"]
            for wanted in OMEGA_COMPONENTS:
                for key in components:
                    if wanted not in key:
                        continue
                    group = components[key]
                    if "Rotation" in group:
                        raw = np.asarray(group["Rotation"][()], dtype=float)
                    else:
                        raw = np.asarray(group.attrs.get("Rotation", []),
                                         dtype=float)
                    raw = raw.ravel()
                    if raw.size != 9:
                        continue
                    rot = raw.reshape(3, 3)
                    angle = float(np.degrees(np.arctan2(rot[0, 2], rot[0, 0])))
                    c, s_ = np.cos(np.radians(angle)), np.sin(np.radians(angle))
                    ideal = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
                    residual = float(np.abs(rot - ideal).max())
                    return sense*angle + offset, angle, residual, key
    except (KeyError, OSError) as trouble:
        print("      omega: could not read the McStas components (%s)" % trouble)
    return None


def replace_sample_information(f_nexus, dg_sample, title=None, sample_name=None,
                               formula=None, omega_deg=None, start_ns=0):
    with h5.File(f_nexus, "r+") as f:
        if omega_deg is not None:
            # entry/instrument/sample_stick_rotation is an NXpositioner; the
            # angle read back from the motor is its NXlog `value`.  One entry,
            # at the run start (time is absolute ns, 'start' = the epoch).
            # The template says 'mm' - wrong for a rotation - so the units
            # are set to 'deg'.
            group = f["entry/instrument/sample_stick_rotation/value"]
            replace_dataset(group, "value", np.array([omega_deg], dtype=float))
            replace_dataset(group, "time", np.array([int(start_ns)]))
            group["value"].attrs["units"] = np.bytes_("deg")
            for name in ("average_value", "minimum_value", "maximum_value"):
                if name in group:
                    replace_dataset(group, name, np.array(omega_deg, dtype=float))
                    group[name].attrs["units"] = np.bytes_("deg")
            print("      sample_stick_rotation = %.4f deg" % omega_deg)
        elif dg_sample is not None and "omega" in dg_sample:
            omega = dg_sample["omega"].to(unit="deg").value
            group = f["entry/instrument/sample_stick_rotation/value"]
            replace_dataset(group, "value", np.array([omega], dtype=float))
            replace_dataset(group, "time", np.array([0]))
            print("      sample_stick_rotation = %.4f deg" % omega)
            for name in ("chi", "phi"):
                if name in dg_sample:
                    value = dg_sample[name].to(unit="deg").value
                    if abs(value) > 1.e-9:
                        print("      NOTE: sample %s = %.4f deg is NOT written - no "
                              "positioner assigned yet" % (name, value))
        # CHANGED 8
        if title is not None:
            set_string(f["entry"], "title", title)
        if sample_name is not None and "name" in f["entry/sample"]:
            set_string(f["entry/sample"], "name", sample_name)
        if formula is not None and "formula" in f["entry/sample"]:
            set_string(f["entry/sample"], "formula", formula)


def set_source_position(f_nexus, dg_magic):
    """Put the McStas flight path into the file.

    CHANGED 12: the CODA template says the source sits at z = -76.55 m, which is
    not this instrument's flight path. McStas has source and sample positions, and
    their distance (159.397 m for these runs) is what every time of flight refers
    to. Mantid puts the sample at the origin, so the source goes to z = -L1.
    """
    source = dg_magic.get("source_position", None)
    sample = dg_magic.get("sample", None)
    if source is None or sample is None or "position" not in sample:
        print("      no source/sample position in the McStas file - keeping the "
              "template value")
        return None
    src = np.asarray(source.value, dtype=float)
    smp = np.asarray(sample["position"].value, dtype=float)
    l1 = float(np.linalg.norm(smp - src))

    with h5.File(f_nexus, "r+") as f:
        path = "entry/instrument/source/transformations/translation"
        if path not in f:
            print("      %s not in the file - source position unchanged" % path)
            return None
        node = f[path]
        vector = np.asarray(node.attrs.get("vector", [0., 0., 1.]), dtype=float)
        old_value = float(np.atleast_1d(node[()]).flat[0])
        # the translation is value * vector; along +z that means value = -L1
        along_z = vector[2] if abs(vector[2]) > 1.e-9 else 1.0
        replace_dataset(f["entry/instrument/source/transformations"], "translation",
                        np.array(-l1 / along_z, dtype=float))
    print("      source position: %.4f m -> %.4f m (L1 from McStas)"
          % (old_value, -l1 / along_z))
    return l1


def update_run_times(f_nexus, start_ns, duration_seconds):
    """CHANGED 3: make /entry/start_time and /entry/end_time match the pulses."""
    end_ns = int(start_ns) + int(duration_seconds * 1e9)
    to_iso = lambda ns: str(np.datetime64(int(ns), "ns")) + "Z"      # noqa: E731
    with h5.File(f_nexus, "r+") as f:
        set_string(f["entry"], "start_time", to_iso(start_ns))
        if "end_time" in f["entry"]:
            set_string(f["entry"], "end_time", to_iso(end_ns))
    print("      run time: %s .. %s" % (to_iso(start_ns), to_iso(end_ns)))


def write_run_number(f_nexus, run_number):
    """Mantid keys peaks by run number, so every file of a scan needs its own."""
    with h5.File(f_nexus, "r+") as f:
        for name in ("run_number", "entry_identifier"):
            if name in f["entry"]:
                del f["entry"][name]
            f["entry"].create_dataset(name, data=np.bytes_(str(run_number)))
    print("      run number %d" % run_number)


def remove_user_info(f_nexus):
    """CHANGED 5: collect first, delete after."""
    with h5.File(f_nexus, "r+") as f:
        names = [name for name in f["entry"]
                 if decode(f["entry"][name].attrs.get("NX_class", "")) == "NXuser"]
        for name in names:
            del f["entry"][name]
    if names:
        print("      removed %d NXuser group(s)" % len(names))


def write_parameters_xml(path, delta_t_ms, delta_l_m, instrument="MAGiC"):
    """Store the refinable corrections as Mantid instrument parameters.

    delta_t (long pulse) and delta_L (flight path) are fitted quantities, so they
    do not belong in the event data. Mantid reads this file next to the IDF and
    the values can be changed without rebuilding anything:

        t0     = ws.getInstrument().getNumberParameter("t0")[0]      # microseconds
        dL     = ws.getInstrument().getNumberParameter("delta_L")[0] # metres
        ws     = ChangeBinOffset(ws, Offset=-t0)
        MoveInstrumentComponent(ws, ComponentName="moderator", Z=-dL,
                                RelativePosition=True)
    """
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<!-- Refinable corrections for %(inst)s, written by mcstas_to_nexus_v2.py.
     Copy next to the IDF or into ~/.mantid/instrument/ -->
<parameter-file instrument="%(inst)s" valid-from="2020-01-01 00:00:00">
  <component-link name="%(inst)s">
    <parameter name="t0"><value val="%(t0).4f"/></parameter>
    <parameter name="delta_L"><value val="%(dl).6f"/></parameter>
  </component-link>
</parameter-file>
""" % {"inst": instrument, "t0": float(delta_t_ms) * 1000.0, "dl": float(delta_l_m)}
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(xml)
    print("  parameters written: %s (t0 = %.1f us, delta_L = %.4f m)"
          % (path, float(delta_t_ms) * 1000.0, float(delta_l_m)))
    return path


# ---------------------------------------------------------------------------
def mcstas_to_coda(mcstas_data_file, template_coda_file, outfile,
                   number_event_detector_a=100000, number_event_detector_b=100000,
                   number_event_cave_monitor=10000, seed=None, poisson=False,
                   mantid_friendly=False, idf=None, title=None,
                   sample_name=None, formula=None, delta_t_ms=3.0,
                   delta_l_m=0.0, apply_delta_t_to_events=False,
                   parameters_xml=None, clear_unfilled=True, n_pulses=1,
                   source_from_mcstas=True, build_idf=None, id_map=None,
                   tof_window=None, assign_by_position=None,
                   omega=None, omega_sense=-1.0, omega_offset=-90.0,
                   pair_planes=True, start_offset_s=0.0, run_number=None):
    print("Writing %s" % outfile)
    rng = np.random.default_rng(seed)
    shutil.copyfile(template_coda_file, outfile)
    start_ns, start_iso = template_start_time(template_coda_file)
    # every run of a scan gets its own time: the pulses, start_time and
    # end_time all move together, so a file is consistent in itself
    start_ns += int(round(float(start_offset_s)*1e9))
    start_iso = str(np.datetime64(start_ns, "ns"))
    print("  pulse time origin: %s (%d ns)" % (start_iso, start_ns))
    print("  pulses requested : %d (%.3f s at %.1f Hz)"
          % (n_pulses, n_pulses / PULSE_FREQUENCY_HZ, PULSE_FREQUENCY_HZ))

    id_maps = load_id_map(id_map)
    position_geometry = None
    if assign_by_position:
        position_geometry = np.load(assign_by_position)
        print("  events will be placed on the mesh of %s, by position"
              % assign_by_position)
        if id_maps:
            print("  --id-map is ignored: the analytic detector_number is not "
                  "used at all in this mode")
            id_maps = {}

    print("Reading McStas file %s ..." % mcstas_data_file)
    dg_magic = read_h5.read_magic_from_nexus(mcstas_data_file)
    da_detector_a = dg_magic.get("detector_a", None)
    da_detector_b = dg_magic.get("detector_b", None)
    da_cave_monitor = dg_magic.get("cave_monitor", None)
    dg_sample = dg_magic.get("sample", None)

    span = 0.0
    for label, da, count in (("a", da_detector_a, number_event_detector_a),
                             ("b", da_detector_b, number_event_detector_b)):
        if da is None:
            print("  detector_%s: NOT in the McStas file" % label)
            if clear_unfilled:
                print("      -> removing the template events so the file does not "
                      "mix measured and simulated data")
                clear_event_group(outfile,
                                  "entry/instrument/magic_detector_%s/"
                                  "magic_detector_%s_event_data" % (label, label))
            else:
                print("      WARNING: the template events of bank %s are kept" % label)
            continue
        print("  converting detector_%s ..." % label)
        data = detector_to_coda_format(
            da, get_det_numbers(outfile, label), count, start_ns, rng, poisson,
            delta_t_ms=delta_t_ms if apply_delta_t_to_events else 0.0,
            n_pulses=n_pulses, tof_window=tof_window,
            position_geometry=position_geometry, label=label)
        if label in id_maps:
            table, id0 = id_maps[label]
            data["event_id"] = apply_id_map(data["event_id"], table, id0)
        replace_detector_event(outfile, data, label)
        span = max(span, data["duration_s"])

    if da_cave_monitor is None and clear_unfilled:
        print("  cave monitor: NOT in the McStas file")
        clear_event_group(outfile,
                          "entry/instrument/beam_monitor_1/beam_monitor_1_events")
    if da_cave_monitor is not None:
        print("  converting the cave monitor ...")
        data = cave_monitor_to_coda_format(
            da_cave_monitor, number_event_cave_monitor, start_ns, rng, poisson,
            delta_t_ms=delta_t_ms if apply_delta_t_to_events else 0.0,
            n_pulses=n_pulses, tof_window=tof_window)
        replace_monitor_event(outfile, data)
        span = max(span, data["duration_s"])

    print("  sample information ...")
    omega_deg = omega
    if omega_deg is None:
        found = omega_from_mcstas(mcstas_data_file, omega_sense, omega_offset)
        if found is None:
            print("      omega: no sample rotation found in the McStas file - "
                  "the log stays empty")
        else:
            omega_deg, raw_angle, residual, component = found
            print("      omega from %s: McStas angle %.4f deg, log value "
                  "%.4f deg (= %+g x angle %+g), residual %.1e%s"
                  % (component, raw_angle, omega_deg, omega_sense,
                     omega_offset, residual,
                     "" if residual < 1e-6 else
                     "  - NOT a pure rotation about the vertical: chi or phi "
                     "is not zero, and one angle does not describe it"))
    else:
        print("      omega given on the command line: %.4f deg" % omega_deg)
    replace_sample_information(outfile, dg_sample, title, sample_name, formula,
                               omega_deg=omega_deg, start_ns=start_ns)
    update_run_times(outfile, start_ns, span)
    if run_number is not None:
        write_run_number(outfile, run_number)
    remove_user_info(outfile)

    if source_from_mcstas:
        print("  source position ...")
        l1 = set_source_position(outfile, dg_magic)
        if l1 and build_idf:
            print("  rebuilding the IDF from the corrected file ...")
            import MAGIC
            if pair_planes:
                # both voxels of a boron layer report the plane between them,
                # which is where the neutron was captured
                MAGIC.PAIR_PLANES = True
                print("      voxel pairs will be put on their boron plane")
            with h5.File(outfile, "r") as f:
                entry = MAGIC.find_entry(f)
                inst = MAGIC.find_instrument(entry)
                banks = [MAGIC.read_bank(f, name, det)
                         for name, det in MAGIC.children(inst, "NXdetector")]
                monitors = MAGIC.collect_monitors(f, entry, inst)
                source_z = MAGIC.source_position(f, inst)
            MAGIC.write_idf(build_idf, banks, monitors, source_z)
            idf = build_idf

    if mantid_friendly:
        print("  making the file Mantid friendly ...")
        try:
            import make_mantid_file
        except ImportError:
            print("      make_mantid_file.py not found next to this script - skipped")
        else:
            with h5.File(outfile, "r+") as f:
                stats = make_mantid_file.sanitize(f)
            print("      fixed-length strings: %d, event_id: %d, offsets: %d"
                  % (stats["strings"], stats["event_id"], stats["offset"]))
            if idf:
                import MAGIC
                MAGIC.embed_idf(outfile, idf)

    if parameters_xml:
        write_parameters_xml(parameters_xml, delta_t_ms, delta_l_m)
    if apply_delta_t_to_events:
        print("  NOTE: delta_t = %.3f ms was baked into the event times" % delta_t_ms)

    print("\nSuccessfully wrote %s (%.0f MB)"
          % (outfile, os.path.getsize(outfile) / 1.e6))
    return outfile


def main(argv=None):
    # line-buffered even through a pipe ("| tee log"), or nothing shows until
    # the end and a 10-minute conversion looks hung
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mcstas", required=True,
                        help="McStas file or glob, e.g. 'fe4o5_*.h5' (quote it)")
    parser.add_argument("--template", required=True,
                        help="CODA file whose NeXus layout is copied")
    parser.add_argument("--outdir", default=".", help="where the .nxs files go")
    parser.add_argument("--suffix", default=".nxs",
                        help="output name = McStas stem + this")
    parser.add_argument("--geometry", default=None, metavar="NPZ",
                        help="voxel mesh for placing events by position")
    parser.add_argument("--idf", default=None,
                        help="IDF to embed in every file (default: built from "
                             "voxelization.py for this batch)")
    parser.add_argument("--build-idf", default=None, metavar="FILE",
                        help="instead: regenerate the IDF from the first file, "
                             "write it here, embed it in all")
    parser.add_argument("--no-pair-planes", dest="pair_planes",
                        action="store_false",
                        help="with --build-idf: voxel centres, not boron planes")
    parser.add_argument("--events-a", type=int, default=10000000)
    parser.add_argument("--events-b", type=int, default=10000000)
    parser.add_argument("--events-monitor", type=int, default=10000)
    parser.add_argument("--pulses", type=int, default=1100)
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the first run; the next ones add 1")
    parser.add_argument("--poisson", action="store_true")
    parser.add_argument("--tof-window", default=None, metavar="LO,HI",
                        help="ms, e.g. 21,71: one 14 Hz frame")
    parser.add_argument("--omega", type=float, default=None,
                        help="write this angle instead of reading it (one run)")
    parser.add_argument("--omega-sense", type=float, default=-1.0)
    parser.add_argument("--omega-offset", type=float, default=-90.0)
    parser.add_argument("--first-run", type=int, default=1,
                        help="run number of the first file; the rest follow")
    parser.add_argument("--title", default=None)
    parser.add_argument("--sample-name", default=None)
    parser.add_argument("--formula", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="read and print the angles, convert nothing")
    args = parser.parse_args(argv)

    files = sorted(glob.glob(args.mcstas))
    if not files:
        print("no McStas file matches %s" % args.mcstas)
        return 1
    print("%d McStas file(s) match %s" % (len(files), args.mcstas))

    print("\nsample rotation, read from each McStas file "
          "(log = %+g x angle %+g):" % (args.omega_sense, args.omega_offset))
    print("  %-26s %-12s %-10s %-10s %s"
          % ("file", "McStas", "log", "residual", "note"))
    angles = {}
    for path in files:
        found = omega_from_mcstas(path, args.omega_sense, args.omega_offset)
        if found is None:
            print("  %-26s no sample rotation found" % os.path.basename(path))
            continue
        value, raw, residual, component = found
        angles[path] = value
        print("  %-26s %-12.3f %-10.3f %-10.1e %s"
              % (os.path.basename(path), raw, value, residual,
                 "about the vertical" if residual < 1e-6 else
                 "NOT a pure omega - chi or phi are non-zero"))
    if args.dry_run:
        print("\ndry run: nothing converted")
        return 0

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    window = ([float(v) for v in args.tof_window.split(",")]
              if args.tof_window else None)
    duration = args.pulses/PULSE_FREQUENCY_HZ
    if not os.path.exists(args.template):
        print("the CODA template %s is not here" % args.template)
        return 1
    print("\nCODA template: %s" % args.template)
    idf = None if args.build_idf else args.idf
    if idf is None and not args.build_idf:
        # section 120: the geometry Mantid uses comes from the same model that
        # gives every event its detector ID, with this batch's segment tilt
        import make_idf_voxelization
        idf = os.path.join(args.outdir, "MAGiC_Definition_vox.xml")
        print("instrument definition from voxelization.py -> %s" % idf)
        if make_idf_voxelization.main(["--mcstas", files[0], "--template",
                                       args.template, "--out", idf]):
            print("building the instrument definition failed")
            return 1
    if idf and not os.path.exists(idf):
        print("the IDF %s is not here" % idf)
        return 1

    results = []
    for number, path in enumerate(files):
        stem = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.outdir, stem + args.suffix)
        run_number = args.first_run + number
        print("\n" + "=" * 70)
        print("[%d/%d] %s -> %s  (run %d)" % (number + 1, len(files),
              os.path.basename(path), os.path.basename(out), run_number))
        if args.skip_existing and os.path.exists(out):
            print("   already there, skipped")
            results.append((out, angles.get(path), "skipped"))
            continue
        try:
            mcstas_to_coda(
                path, args.template, out,
                number_event_detector_a=args.events_a,
                number_event_detector_b=args.events_b,
                number_event_cave_monitor=args.events_monitor,
                seed=args.seed + number, poisson=args.poisson,
                mantid_friendly=True, idf=idf,
                title=args.title, sample_name=args.sample_name,
                formula=args.formula, parameters_xml=None,
                n_pulses=args.pulses, tof_window=window,
                assign_by_position=args.geometry,
                build_idf=args.build_idf if idf is None else None,
                pair_planes=args.pair_planes,
                omega=args.omega, omega_sense=args.omega_sense,
                omega_offset=args.omega_offset,
                start_offset_s=number*(duration + 1.0),
                run_number=run_number)
            if idf is None:                 # built from the first file
                idf = args.build_idf
            results.append((out, args.omega if args.omega is not None
                            else angles.get(path), "ok"))
        except Exception as exc:                  # one bad run must not
            print("   FAILED: %s" % exc)          # cost the other thirty
            traceback.print_exc(limit=3)
            results.append((out, angles.get(path), "failed"))

    print("\n" + "=" * 70)
    print("summary")
    print("  %-34s %-10s %s" % ("file", "omega", "state"))
    for out, omega, state in results:
        print("  %-34s %-10s %s" % (os.path.basename(out),
              "-" if omega is None else "%.3f" % omega, state))
    good = sum(1 for _, _, s in results if s in ("ok", "skipped"))
    print("  %d of %d usable" % (good, len(results)))
    return 0 if good == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
