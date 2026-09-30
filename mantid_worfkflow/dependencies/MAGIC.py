#!/usr/bin/env python3
"""
MAGIC.py - generate a Mantid Instrument Definition File (IDF) for MAGiC (ESS)
from the voxel geometry stored in an ESS NeXus event file (e.g. mccode.nxs).

What it does
    1. reads every NXdetector of the file (MAGiC has two banks: detector_a, detector_b)
    2. gets the voxel centres either from x/y/z_pixel_offset or from the NXoff_geometry
       mesh (vertices / winding_order / faces / detector_faces), and the detector IDs
       from detector_number
    3. applies the full depends_on / NXtransformations chain of each bank, so the voxels
       end up in the Mantid laboratory frame (beam along +z, y up, sample at the origin)
    4. writes <INSTRUMENT>_Definition.xml: source, sample, monitors and one <detector>
       component per voxel, with IDs identical to the event_id values in the file
    5. optionally embeds that IDF into a copy of the NeXus file, so that
       LoadEventNexus uses it instead of the NeXus geometry (--embed)

Usage
    python3 MAGIC.py --report                        # just describe the file
    python3 MAGIC.py                                 # write MAGiC_Definition.xml
    python3 MAGIC.py --embed mccode_idf.nxs          # ... and embed it into a copy
    python3 MAGIC.py --voxel-size 0.01,0.01,0.01     # override the voxel shape
    python3 MAGIC.py --with-geom-xml                 # also write the old SNS geometry XML

Run it with any python that has numpy + h5py - for example the Mantid Workbench
script editor.
"""

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------
INST_NAME = "MAGiC"            # must match /entry/instrument/name in the NeXus file
NXS_FILE = "mccode.nxs"
IDF_FILE = "%s_Definition.xml" % INST_NAME
DEFAULT_L1 = -160.0            # m, fallback source position if the file has no NXsource
VALID_FROM = "2020-01-01 00:00:00"

LENGTH_UNITS = {"m": 1.0, "metre": 1.0, "meter": 1.0, "meters": 1.0,
                "mm": 1.0e-3, "millimetre": 1.0e-3, "millimeter": 1.0e-3,
                "cm": 1.0e-2, "centimetre": 1.0e-2, "centimeter": 1.0e-2,
                "um": 1.0e-6, "micrometre": 1.0e-6}
ANGLE_UNITS = {"deg": np.pi / 180., "degree": np.pi / 180., "degrees": np.pi / 180.,
               "rad": 1.0, "radian": 1.0, "radians": 1.0}


# ---------------------------------------------------------------------------
# small NeXus helpers (duck typed: anything with .keys() is treated as a group)
# ---------------------------------------------------------------------------
def decode(value):
    """bytes / numpy bytes -> str, everything else unchanged."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.dtype.kind in "SU" and value.size == 1:
        return decode(value.flat[0])
    return value


def is_group(obj):
    return hasattr(obj, "keys")


def nx_class(obj):
    return str(decode(obj.attrs.get("NX_class", "")))


def attr(obj, name, default=None):
    if name not in obj.attrs:
        return default
    return decode(obj.attrs[name])


def children(group, nxclass=None):
    """(name, object) pairs of a group, optionally filtered by NX_class."""
    out = []
    for name in group.keys():
        obj = group[name]
        if nxclass is None or nx_class(obj) == nxclass:
            out.append((name, obj))
    return sorted(out, key=lambda item: item[0])


def walk(group, nxclass=None, prefix=""):
    """Recursively yield (path, group) pairs, optionally filtered by NX_class."""
    for name, obj in children(group):
        if not is_group(obj):
            continue
        path = prefix + "/" + name
        if nxclass is None or nx_class(obj) == nxclass:
            yield path, obj
        else:
            for item in walk(obj, nxclass, path):
                yield item


def find_entry(handle):
    """Return the NXentry group (usually /entry, ISIS files use /raw_data_1)."""
    for name in handle.keys():
        obj = handle[name]
        if is_group(obj) and nx_class(obj) == "NXentry":
            return obj
    for name in ("entry", "raw_data_1"):
        if name in handle:
            return handle[name]
    raise KeyError("No NXentry group found in the file")


def find_instrument(entry):
    for _name, obj in children(entry, "NXinstrument"):
        return obj
    if "instrument" in entry:
        return entry["instrument"]
    raise KeyError("No NXinstrument group found in %s" % entry.name)


def scalar(obj):
    """Value of a transformation field: plain dataset, or NXlog with a 'value' array."""
    if is_group(obj):                      # NXlog / NXpositioner
        for key in ("value", "average_value", "raw_value"):
            if key in obj:
                return scalar(obj[key])
        raise KeyError("No value in log %s" % obj.name)
    data = obj[()]
    data = np.atleast_1d(np.asarray(data))
    if data.size == 0:
        return 0.0
    return float(data.flat[0])


# ---------------------------------------------------------------------------
# NXtransformations
# ---------------------------------------------------------------------------
def transformation_matrix(node):
    """4x4 matrix of one NXtransformations field (translation or rotation)."""
    ttype = str(attr(node, "transformation_type", "")).lower()
    vector = np.asarray(attr(node, "vector", [0., 0., 1.]), dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        vector = np.array([0., 0., 1.])
    else:
        vector = vector / norm
    units = str(attr(node, "units", "")).lower()
    value = scalar(node)

    matrix = np.eye(4)
    if ttype == "translation":
        matrix[:3, 3] = vector * value * LENGTH_UNITS.get(units, 1.0)
    elif ttype == "rotation":
        angle = value * ANGLE_UNITS.get(units, np.pi / 180.)
        c, s = np.cos(angle), np.sin(angle)
        x, y, z = vector
        # right handed rotation about `vector` (Rodrigues)
        matrix[:3, :3] = np.array([
            [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
            [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
            [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]])
    else:
        raise ValueError("Unknown transformation_type '%s' in %s"
                         % (ttype, getattr(node, "name", "?")))

    # a constant offset may be attached to the field
    offset = attr(node, "offset", None)
    if offset is not None:
        off_units = str(attr(node, "offset_units", units)).lower()
        matrix[:3, 3] = matrix[:3, 3] + np.asarray(offset, dtype=float) \
            * LENGTH_UNITS.get(off_units, 1.0)
    return matrix


def resolve_path(handle, group, path):
    """depends_on strings can be absolute or relative to the group holding them."""
    path = str(decode(path))
    if path in (".", ""):
        return None
    if path.startswith("/"):
        return handle[path]
    return group[path]


def total_transform(handle, group, verbose=False):
    """Full local -> laboratory 4x4 matrix of a component (follows depends_on)."""
    matrix = np.eye(4)
    if "depends_on" not in group:
        return matrix
    node = resolve_path(handle, group, group["depends_on"][()])
    parent = group
    guard = 0
    while node is not None:
        step = transformation_matrix(node)
        # the field closest to the component acts first: M_total = M_n ... M_2 M_1
        matrix = step @ matrix
        if verbose:
            print("      %-60s -> %s" % (getattr(node, "name", "?"),
                                         np.round(step[:3, 3], 4)))
        nxt = attr(node, "depends_on", ".")
        parent = node.parent if hasattr(node, "parent") else parent
        node = resolve_path(handle, parent, nxt)
        guard += 1
        if guard > 32:
            raise RuntimeError("depends_on chain too deep - is it circular?")
    return matrix


def apply_transform(matrix, points):
    """Apply a 4x4 matrix to an (N,3) array of points."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def component_position(handle, group):
    """Laboratory position (metres) of a component with no pixels of its own."""
    return apply_transform(total_transform(handle, group), np.zeros((1, 3)))[0]


# ---------------------------------------------------------------------------
# detector voxels
# ---------------------------------------------------------------------------
def off_voxel_centres(off, detector_number):
    """Voxel centres and voxel size from an NXoff_geometry mesh.

    vertices       (Nv,3)  mesh vertices
    winding_order  (Nw,)   vertex indices, faces are slices of this array
    faces          (Nf,)   start index of each face inside winding_order
    detector_faces (Nf,2)  [face index, detector id]
    """
    vertices = np.asarray(off["vertices"][()], dtype=float)
    scale = LENGTH_UNITS.get(str(attr(off["vertices"], "units", "m")).lower(), 1.0)
    vertices = vertices * scale
    winding = np.asarray(off["winding_order"][()]).ravel().astype(np.int64)
    faces = np.asarray(off["faces"][()]).ravel().astype(np.int64)
    det_faces = np.asarray(off["detector_faces"][()]).reshape(-1, 2).astype(np.int64)

    # vertex index -> detector id, via the face each vertex belongs to
    face_len = np.diff(np.append(faces, winding.size))
    face_of_vertexslot = np.repeat(np.arange(faces.size), face_len)

    face_to_det = np.full(faces.size, -1, dtype=np.int64)
    face_to_det[det_faces[:, 0]] = det_faces[:, 1]
    det_of_vertexslot = face_to_det[face_of_vertexslot]

    keep = det_of_vertexslot >= 0
    det_ids = det_of_vertexslot[keep]
    coords = vertices[winding[keep]]

    order = np.argsort(det_ids, kind="stable")
    det_ids, coords = det_ids[order], coords[order]
    uniq, start = np.unique(det_ids, return_index=True)
    counts = np.diff(np.append(start, det_ids.size))

    centres = np.vstack([np.add.reduceat(coords[:, i], start) / counts
                         for i in range(3)]).T

    # voxel size: median bounding box of the individual voxels (sampled)
    sample = np.arange(uniq.size)
    if uniq.size > 500:
        sample = np.linspace(0, uniq.size - 1, 500).astype(int)
    extents = []
    for i in sample:
        block = coords[start[i]:start[i] + counts[i]]
        extents.append(block.max(axis=0) - block.min(axis=0))
    size = np.median(np.array(extents), axis=0)
    size = np.where(size > 1.e-6, size, 0.002)

    if detector_number is not None:
        wanted = np.asarray(detector_number).ravel().astype(np.int64)
        lookup = {int(d): i for i, d in enumerate(uniq)}
        missing = [d for d in wanted if int(d) not in lookup]
        if missing:
            print("      WARNING: %d detector_number entries have no face in the mesh "
                  "(e.g. %s) - they are skipped" % (len(missing), missing[:5]))
        index = np.array([lookup[int(d)] for d in wanted if int(d) in lookup])
        uniq = uniq[index]
        centres = centres[index]
    return uniq, centres, size


def pixel_offset_centres(det, detector_number):
    """Voxel centres from x/y/z_pixel_offset."""
    def read(name):
        if name not in det:
            return None
        data = np.asarray(det[name][()], dtype=float).ravel()
        scale = LENGTH_UNITS.get(str(attr(det[name], "units", "m")).lower(), 1.0)
        return data * scale

    x = read("x_pixel_offset")
    y = read("y_pixel_offset")
    z = read("z_pixel_offset")
    if x is None or y is None:
        return None
    if z is None:
        z = np.zeros_like(x)
    centres = np.vstack([x, y, z]).T
    ids = np.asarray(detector_number).ravel().astype(np.int64)
    if ids.size != centres.shape[0]:
        raise ValueError("detector_number (%d) and pixel offsets (%d) disagree"
                         % (ids.size, centres.shape[0]))
    return ids, centres, np.array([0.005, 0.005, 0.005])


# Both voxels of a boron-layer pair report the same place. ON by default since
# the angular budget was measured properly: using voxel centres instead of the
# plane adds a systematic +-0.227 deg, alternating in sign within a pair, while
# the instrument's whole angular resolution is 0.10 deg RMS (anode 0.027,
# cathode 0.015, 4 mm sample 0.081). So the voxel-centre convention is not a
# small residual - it more than doubles the angular uncertainty and is the
# largest single term. --no-pair-planes restores the old behaviour for
# comparison.
PAIR_PLANES = True


def boron_plane_positions(pos, name="", report=True):
    """Put both voxels of a boron-layer pair on the plane they share.

    How the detector works. A neutron is captured on a layer of boron, and a gas
    voxel sits on each side of it. Which of the two fires is what the electronics
    report, but the neutron was absorbed on the boron - so the position that
    belongs to an event is the plane, not the centre of either voxel.

    The mesh in the CODA file gives voxel centres, and they come in pairs: along
    a row the angular step alternates between a short one, inside a pair, and a
    longer one, from one boron layer to the next. Bank A: 0.4827 and 0.5673 deg,
    1.05 deg per layer. Bank B: 0.4551 and 0.4849, 0.94 deg per layer. Those sums
    are exactly the delta_gamma_vs of the two banks in voxelization.py, and the
    midpoints of the short-step pairs reproduce that model's boron planes to
    0.033 deg in bank A and 0.002 deg in bank B, with no scatter.

    So both voxels of a pair are moved to their midpoint. The shift is about
    5.4 mm, which is 0.24 deg at this distance - 0.015 A^-1 at 1.75 A and
    0.042 A^-1 at 0.62 A. That was always within the indexing budget, so this is
    not a fix for a broken chain; it is a description of the instrument that
    matches how it actually records a neutron.

    The pairing is detected from the positions, not assumed.
    """
    y = pos[:, 1]
    breaks = np.flatnonzero(np.abs(np.diff(y)) > 1.e-6) + 1
    runs = np.diff(np.concatenate(([0], breaks, [pos.shape[0]])))
    if runs.size < 2 or not np.all(runs == runs[0]):
        print("    %-22s no row structure, boron planes not applied" % name)
        return pos
    width = int(runs[0])
    if width < 4 or width % 2:
        print("    %-22s row length %d is not an even number, boron planes "
              "not applied" % (name, width))
        return pos

    rows = pos.reshape(-1, width, 3).copy()
    step = np.linalg.norm(np.diff(rows[0], axis=0), axis=1)
    short_first = step[0::2].mean() < step[1::2].mean()
    offset = 0 if short_first else 1
    inside = step[offset::2].mean()
    between = step[1 - offset::2].mean()
    if inside >= between:
        print("    %-22s the two steps are not distinguishable, boron planes "
              "not applied" % name)
        return pos

    first = offset
    last = first + 2 * ((width - first) // 2)
    block = rows[:, first:last, :].reshape(rows.shape[0], -1, 2, 3)
    middle = block.mean(axis=2, keepdims=True)
    rows[:, first:last, :] = np.repeat(middle, 2, axis=2).reshape(
        rows.shape[0], last - first, 3)
    moved = pos.reshape(-1, width, 3) - rows
    shift = np.linalg.norm(moved.reshape(-1, 3), axis=1)
    radius = np.linalg.norm(pos, axis=1)
    if report:
        print("    %-22s boron planes: pairs start at index %d, step inside "
              "%.4f m, between %.4f m" % (name, offset, inside, between))
        print("    %-22s   %d voxels moved by a median %.4f m = %.3f deg; "
              "%d left as they are"
              % ("", int((shift > 0).sum()), float(np.median(shift[shift > 0])),
                 float(np.degrees(np.median(shift[shift > 0] / radius[shift > 0]))),
                 int((shift == 0).sum())))
    return rows.reshape(pos.shape)


def read_bank(handle, name, det, verbose=False):
    """Return dict(name, ids, pos (lab frame, metres), size)."""
    if "detector_number" not in det:
        raise KeyError("%s has no detector_number" % name)
    detector_number = np.asarray(det["detector_number"][()]).ravel().astype(np.int64)

    result = pixel_offset_centres(det, detector_number)
    if result is not None:
        ids, local, size = result
        source = "pixel offsets"
    else:
        off = None
        for child_name, obj in children(det):
            if is_group(obj) and nx_class(obj) == "NXoff_geometry" \
                    and "detector_faces" in obj:
                off = obj
                break
        if off is None:
            raise KeyError("%s has neither pixel offsets nor an NXoff_geometry with "
                           "detector_faces" % name)
        ids, local, size = off_voxel_centres(off, detector_number)
        source = "NXoff_geometry (%s)" % off.name.split("/")[-1]

    matrix = total_transform(handle, det, verbose=verbose)
    pos = apply_transform(matrix, local)
    if PAIR_PLANES:
        pos = boron_plane_positions(pos, name)

    bank = {"name": name, "ids": ids, "pos": pos, "size": size, "source": source}
    describe_bank(bank)
    return bank


def describe_bank(bank):
    pos, ids = bank["pos"], bank["ids"]
    radius = np.linalg.norm(pos, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        two_theta = np.degrees(np.arccos(np.clip(pos[:, 2] / radius, -1., 1.)))
    print("    %-22s %7d voxels, ids %d..%d, from %s"
          % (bank["name"], ids.size, ids.min(), ids.max(), bank["source"]))
    print("      L2       %.4f .. %.4f m" % (radius.min(), radius.max()))
    print("      2theta   %.2f .. %.2f deg" % (two_theta.min(), two_theta.max()))
    print("      centre   x=%.4f y=%.4f z=%.4f m" % tuple(pos.mean(axis=0)))
    print("      voxel    %.2f x %.2f x %.2f mm" % tuple(1000 * np.asarray(bank["size"])))


# ---------------------------------------------------------------------------
# IDF
# ---------------------------------------------------------------------------
def cuboid_xml(name, size, tag="detector"):
    (dx, dy, dz) = [0.5 * float(v) for v in size]
    return """  <type name="{name}" is="{tag}">
    <cuboid id="{name}-shape">
      <left-front-bottom-point  x="{mx:.6f}" y="{my:.6f}" z="{mz:.6f}"/>
      <left-front-top-point     x="{mx:.6f}" y="{py:.6f}" z="{mz:.6f}"/>
      <left-back-bottom-point   x="{mx:.6f}" y="{my:.6f}" z="{pz:.6f}"/>
      <right-front-bottom-point x="{px:.6f}" y="{my:.6f}" z="{mz:.6f}"/>
    </cuboid>
    <algebra val="{name}-shape"/>
  </type>
""".format(name=name, tag=tag, mx=-dx, px=dx, my=-dy, py=dy, mz=-dz, pz=dz)


def idlist_xml(idname, ids):
    """Explicit ID list, in the same order as the <component> elements.
    Consecutive runs are written as ranges to keep the file small."""
    lines = ['  <idlist idname="%s">' % idname]
    ids = np.asarray(ids, dtype=np.int64)
    start = 0
    for i in range(1, ids.size + 1):
        if i < ids.size and ids[i] == ids[i - 1] + 1:
            continue
        if i - start >= 3:
            lines.append('    <id start="%d" end="%d"/>' % (ids[start], ids[i - 1]))
        else:
            lines.extend('    <id val="%d"/>' % v for v in ids[start:i])
        start = i
    lines.append("  </idlist>")
    return "\n".join(lines) + "\n"


def bank_xml(bank):
    """<component> + <type> holding one voxel component per detector ID."""
    name = bank["name"]
    voxel = "%s_voxel" % name
    head = ('  <component type="%s" idlist="%s_ids" name="%s">\n'
            '    <location/>\n  </component>\n\n  <type name="%s">\n'
            % (name, name, name, name))
    body = "".join(
        '    <component type="%s"><location x="%.6f" y="%.6f" z="%.6f"/></component>\n'
        % (voxel, x, y, z) for (x, y, z) in bank["pos"])
    return head + body + "  </type>\n\n" + cuboid_xml(voxel, bank["size"]) + "\n"


def detect_structure(pos, y_tol=1.e-6, xz_tol=1.e-6):
    """Find the (segment, row, voxel) lattice of a bank.

    MAGiC banks are built from segments; inside a segment every row has the same
    (x, z) pattern and only y changes. Detecting that turns a 500 000 line flat
    IDF into a few thousand lines. Returns (rows_per_segment, voxels_per_row) or
    None when the pattern does not hold.
    """
    n = pos.shape[0]
    y = pos[:, 1]
    breaks = np.flatnonzero(np.abs(np.diff(y)) > y_tol) + 1
    runs = np.diff(np.concatenate(([0], breaks, [n])))
    if runs.size < 2 or not np.all(runs == runs[0]):
        return None
    voxels_per_row = int(runs[0])
    if n % voxels_per_row:
        return None

    rows = pos.reshape(-1, voxels_per_row, 3)
    xz = rows[:, :, [0, 2]]
    changed = np.abs(xz[1:] - xz[:-1]).reshape(rows.shape[0] - 1, -1).max(axis=1) > xz_tol
    seg_breaks = np.flatnonzero(changed) + 1
    seg_runs = np.diff(np.concatenate(([0], seg_breaks, [rows.shape[0]])))
    if seg_runs.size < 2 or not np.all(seg_runs == seg_runs[0]):
        return None
    return int(seg_runs[0]), voxels_per_row


def bank_xml_structured(bank, rows_per_segment, voxels_per_row):
    """Compact IDF: bank -> segments -> rows -> voxels.

    Detector IDs are assigned by Mantid in tree order, which is exactly the order
    of detector_number, so a single <id start=.. end=..> list is enough.
    """
    name = bank["name"]
    voxel = "%s_voxel" % name
    pos = bank["pos"]
    n_rows = pos.shape[0] // voxels_per_row
    n_seg = n_rows // rows_per_segment
    rows = pos.reshape(n_rows, voxels_per_row, 3)

    out = ['  <component type="%s" idlist="%s_ids" name="%s">\n    <location/>\n'
           '  </component>\n\n  <type name="%s">\n' % (name, name, name, name)]
    out.extend('    <component type="%s_seg%d"><location/></component>\n' % (name, s)
               for s in range(n_seg))
    out.append("  </type>\n\n")

    for s in range(n_seg):
        out.append('  <type name="%s_seg%d">\n' % (name, s))
        for r in range(s * rows_per_segment, (s + 1) * rows_per_segment):
            out.append('    <component type="%s_row%d"><location y="%.6f"/>'
                       '</component>\n' % (name, s, rows[r, 0, 1]))
        out.append("  </type>\n")
        # one row type per segment: the (x, z) pattern shared by its rows
        out.append('  <type name="%s_row%d">\n' % (name, s))
        for (x, _y, z) in rows[s * rows_per_segment]:
            out.append('    <component type="%s"><location x="%.6f" z="%.6f"/>'
                       '</component>\n' % (voxel, x, z))
        out.append("  </type>\n\n")

    out.append(cuboid_xml(voxel, bank["size"]))
    return "".join(out)


def monitor_xml(monitors):
    if not monitors:
        return ""
    body = "".join(
        '    <component type="monitor"><location x="%.6f" y="%.6f" z="%.6f" '
        'name="%s"/></component>\n' % (m["pos"][0], m["pos"][1], m["pos"][2], m["name"])
        for m in monitors)
    ids = [m["id"] for m in monitors]
    return ('  <component type="monitors" idlist="monitors"><location/></component>\n'
            '  <type name="monitors">\n' + body + "  </type>\n\n"
            + cuboid_xml("monitor", (0.04, 0.04, 0.01), tag="monitor")
            + idlist_xml("monitors", ids) + "\n")


def write_idf(filename, banks, monitors, source_z, inst_name=INST_NAME,
               flat=False):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = ["""<?xml version="1.0" encoding="UTF-8"?>
<!-- Mantid instrument definition file for {inst}, generated by MAGIC.py on {now}
     Voxel positions and detector IDs come straight from the NeXus file, so they
     match the event_id values in the data. -->
<instrument xmlns="http://www.mantidproject.org/IDF/1.0"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://www.mantidproject.org/IDF/1.0 http://schema.mantidproject.org/IDF/1.0/IDFSchema.xsd"
            name="{inst}" valid-from="{valid}" valid-to="2100-12-31 23:59:59"
            last-modified="{now}">

  <defaults>
    <length unit="metre"/>
    <angle unit="degree"/>
    <reference-frame>
      <along-beam axis="z"/>
      <pointing-up axis="y"/>
      <handedness val="right"/>
      <theta-sign axis="x"/>
    </reference-frame>
    <default-view view="spherical_y"/>
  </defaults>

  <!--SOURCE-->
  <component type="moderator"><location z="{source_z:.4f}"/></component>
  <type name="moderator" is="Source"/>

  <!--SAMPLE-->
  <component type="sample-position"><location x="0.0" y="0.0" z="0.0"/></component>
  <type name="sample-position" is="SamplePos"/>

""".format(inst=inst_name, now=now, valid=VALID_FROM, source_z=source_z)]

    if monitors:
        parts.append("  <!--MONITORS-->\n" + monitor_xml(monitors))

    for bank in banks:
        parts.append("  <!--DETECTOR BANK %s-->\n" % bank["name"])
        shape = None if flat else detect_structure(bank["pos"])
        if shape is None:
            parts.append(bank_xml(bank))
            print("    %s: flat list of %d voxels" % (bank["name"], bank["ids"].size))
        else:
            rows_per_segment, voxels_per_row = shape
            n_rows = bank["pos"].shape[0] // voxels_per_row
            print("    %s: %d segments x %d rows x %d voxels"
                  % (bank["name"], n_rows // rows_per_segment, rows_per_segment,
                     voxels_per_row))
            parts.append(bank_xml_structured(bank, rows_per_segment, voxels_per_row))
    for bank in banks:
        parts.append(idlist_xml("%s_ids" % bank["name"], bank["ids"]))
    parts.append("\n</instrument>\n")

    xml = "".join(parts)
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write(xml)
    ET.parse(filename)                      # fails loudly if the XML is malformed
    total = sum(b["ids"].size for b in banks)
    print("  IDF written: %s (%.1f kB, %d voxels in %d bank(s))"
          % (filename, os.path.getsize(filename) / 1024., total, len(banks)))
    return filename


# ---------------------------------------------------------------------------
# embedding the IDF into a NeXus file (same as simple_tof_example/embed_idf.py)
# ---------------------------------------------------------------------------
def embed_idf(nxs_in, idf_path, nxs_out=None):
    import h5py
    with open(idf_path, "r", encoding="utf-8") as handle:
        xml_text = handle.read()
    idf_name = ET.fromstring(xml_text.encode("utf-8")).get("name")

    target = nxs_in
    if nxs_out is not None and os.path.abspath(nxs_out) != os.path.abspath(nxs_in):
        print("  copying %s -> %s ..." % (nxs_in, nxs_out))
        shutil.copyfile(nxs_in, nxs_out)
        target = nxs_out

    with h5py.File(target, "r+") as handle:
        inst = find_instrument(find_entry(handle))
        if "name" in inst:
            current = str(decode(inst["name"][()])).strip()
            if current != idf_name:
                print("  WARNING: /instrument/name = '%s' but the IDF is named '%s'"
                      " - replacing it" % (current, idf_name))
                del inst["name"]
                inst.create_dataset("name", data=np.bytes_(idf_name.encode()))
        else:
            inst.create_dataset("name", data=np.bytes_(idf_name.encode()))

        if "instrument_xml" in inst:
            del inst["instrument_xml"]
        note = inst.create_group("instrument_xml")
        note.attrs["NX_class"] = np.bytes_(b"NXnote")
        note.create_dataset("data", data=np.bytes_(xml_text.encode("utf-8")))
        note.create_dataset("type", data=np.bytes_(b"text/xml"))
        note.create_dataset("description",
                            data=np.bytes_(b"XML contents of the instrument IDF"))
    print("  embedded the IDF into %s (LoadEventNexus will now use it)" % target)
    return target


# ---------------------------------------------------------------------------
# report mode
# ---------------------------------------------------------------------------
def report(handle):
    entry = find_entry(handle)
    inst = find_instrument(entry)
    print("  entry      : %s" % entry.name)
    if "name" in inst:
        print("  instrument : %s" % decode(inst["name"][()]))
    for name, obj in children(inst):
        cls = nx_class(obj)
        if cls in ("NXdetector", "NXsource", "NXmonitor", "NXdisk_chopper", "NXslit"):
            try:
                pos = component_position(handle, obj)
                where = "at x=%.3f y=%.3f z=%.3f m" % tuple(pos)
            except Exception as err:                       # noqa: BLE001
                where = "(no position: %s)" % err
            extra = ""
            if cls == "NXdetector" and "detector_number" in obj:
                det_num = np.asarray(obj["detector_number"][()]).ravel()
                extra = ", %d detector_number (%d..%d)" % (det_num.size,
                                                           det_num.min(), det_num.max())
                sub = [n for n, o in children(obj) if is_group(o)]
                extra += ", groups: %s" % ", ".join(sub)
            print("  %-26s %-16s %s%s" % (name, cls, where, extra))
    for path, obj in walk(entry, "NXevent_data"):
        keys = ", ".join(sorted(obj.keys()))
        size = obj["event_id"].size if "event_id" in obj else 0
        print("  %-26s %-16s %d events (%s)" % (path, "NXevent_data", size, keys))


def check_event_ids(entry, banks, sample=1000000):
    """Compare the detector IDs of the IDF with the event_id values in the file."""
    known = np.unique(np.concatenate([b["ids"] for b in banks]))
    print("  checking event_id values against the %d detector IDs of the IDF"
          % known.size)
    for path, group in walk(entry, "NXevent_data"):
        if "event_id" not in group:
            continue
        data = group["event_id"]
        total = int(getattr(data, "size", 0) or np.asarray(data[()]).size)
        ids = np.asarray(data[0:min(total, sample)]).ravel()
        if ids.size == 0:
            print("    %-40s empty" % path)
            continue
        unknown = np.setdiff1d(np.unique(ids), known)
        status = "OK" if unknown.size == 0 else \
            "%d unknown id(s), e.g. %s" % (unknown.size, unknown[:5].tolist())
        print("    %-40s %d events (checked %d), ids %d..%d -> %s"
              % (path, total, ids.size, ids.min(), ids.max(), status))


def collect_monitors(handle, entry, inst):
    monitors, next_id = [], -1
    for parent in (entry, inst):
        for name, obj in children(parent, "NXmonitor"):
            try:
                pos = component_position(handle, obj)
            except Exception:                              # noqa: BLE001
                continue
            mon_id = None
            if "detector_id" in obj:
                mon_id = int(np.asarray(obj["detector_id"][()]).ravel()[0])
            else:
                # ESS files carry the monitor ID only inside its event data
                for _path, ev in walk(obj, "NXevent_data"):
                    if "event_id" in ev and ev["event_id"].size:
                        mon_id = int(np.rint(np.asarray(ev["event_id"][0:1]).ravel()[0]))
                        break
            if mon_id is None:
                mon_id = next_id
                next_id -= 1
            monitors.append({"name": name, "pos": pos, "id": mon_id})
    return monitors


def source_position(handle, inst):
    for _name, obj in children(inst, "NXsource"):
        try:
            return float(component_position(handle, obj)[2])
        except Exception as err:                           # noqa: BLE001
            print("  WARNING: cannot place the NXsource (%s)" % err)
    print("  WARNING: no usable NXsource, using z = %.1f m" % DEFAULT_L1)
    return DEFAULT_L1


# ---------------------------------------------------------------------------
# optional: the old SNS-style geometry XML (ess_geometry.py)
# ---------------------------------------------------------------------------
def write_geom_xml(banks, monitors, source_z, inst_name=INST_NAME):
    """Legacy SNS DAS geometry file - kept for reference, not used by Mantid."""
    from ess_geometry import Geometry, Component, Maths, Recipe

    geometry = Geometry(inst_name)
    try:
        from ess_geometry import generateGeom
        geometry = generateGeom(inst_name)
        entry = geometry.getEntry()
        instrument = entry.getInstrument()
    except Exception as err:                               # noqa: BLE001
        print("  WARNING: generateGeom(%s) failed (%s), building the tree by hand"
              % (inst_name, err))
        from ess_geometry import Entry, Instrument
        entry = Entry(inst_name)
        instrument = Instrument()
        entry.addInstrument(instrument)
        geometry.addEntry(entry)

    sample = Component("sample", "NXsample")
    sample.setComment(" SAMPLE ")
    sample.setHelper("Goiniometer")
    for var in ("phi", "chi", "omega"):
        sample.addVariable(var, "real_%s" % var)
    entry.addSample(sample)

    for mon in monitors:
        component = Component(mon["name"], "NXmonitor")
        component.setHelper("ParameterCopy")
        component.addParameter("distance", "%.4f" % float(np.linalg.norm(mon["pos"])),
                               units="metre")
        entry.addMonitor(component)

    for bank in banks:
        component = Component(bank["name"], "NXdetector")
        component.setRecipe("magic_detector")
        component.addParameter("numberOfVoxels", "%d" % bank["ids"].size)
        component.addParameter("cenDistance",
                               "%.4f" % float(np.linalg.norm(bank["pos"].mean(axis=0))),
                               units="metre")
        component.setAnnotation("<local_name>%s</local_name>" % bank["name"])
        instrument.addComponent(component)

    recipe = Recipe("magic_detector")
    recipe.setHelper("Rectangle")
    geometry.addRecipe(recipe)

    maths = Maths()
    maths.addDefinition("L1", "%.4f" % abs(source_z), units="metre")
    maths.addEquation("a=L1")
    maths.addOutput("L1", "length")
    geometry.addMath(maths)

    name = geometry.writeToFile()
    print("  SNS-style geometry written: %s" % name)


# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nxs", default=NXS_FILE, help="input NeXus file")
    parser.add_argument("--out", default=IDF_FILE, help="IDF file to write")
    parser.add_argument("--name", default=INST_NAME, help="instrument name")
    parser.add_argument("--embed", metavar="FILE", default=None,
                        help="also write a copy of the NeXus file with the IDF inside")
    parser.add_argument("--banks", default=None,
                        help="comma separated NXdetector names (default: all)")
    parser.add_argument("--voxel-size", default=None, metavar="DX,DY,DZ",
                        help="override the voxel size in metres")
    parser.add_argument("--no-event-check", action="store_true",
                        help="skip comparing the detector IDs with the event_id values")
    parser.add_argument("--pair-planes", action="store_true", default=True,
                        help="put both voxels of a boron-layer pair on the "
                             "plane they share, which is where the neutron was "
                             "actually captured (default; see "
                             "boron_plane_positions)")
    parser.add_argument("--no-pair-planes", dest="pair_planes",
                        action="store_false",
                        help="use voxel centres instead, for comparison")
    parser.add_argument("--flat-idf", action="store_true",
                        help="write one <component> per voxel instead of the compact "
                             "segment/row hierarchy")
    parser.add_argument("--report", action="store_true",
                        help="only describe the NeXus file and exit")
    parser.add_argument("--with-geom-xml", action="store_true",
                        help="also write the legacy SNS geometry XML")
    parser.add_argument("--verbose", action="store_true",
                        help="print every transformation of the depends_on chains")
    args = parser.parse_args(argv)
    globals()["PAIR_PLANES"] = bool(getattr(args, "pair_planes", True))

    try:
        import h5py
    except ImportError:
        raise SystemExit("h5py is required - run this inside Mantid Workbench, "
                         "or 'pip install h5py'")

    if not os.path.exists(args.nxs):
        raise SystemExit("No such file: %s" % args.nxs)

    print("Reading %s" % args.nxs)
    with h5py.File(args.nxs, "r") as handle:
        if args.report:
            report(handle)
            return 0

        entry = find_entry(handle)
        inst = find_instrument(entry)

        wanted = None
        if args.banks:
            wanted = [b.strip() for b in args.banks.split(",")]
        banks = []
        for name, det in children(inst, "NXdetector"):
            if wanted is not None and name not in wanted:
                continue
            banks.append(read_bank(handle, name, det, verbose=args.verbose))
        if not banks:
            raise SystemExit("No NXdetector groups found (try --report)")

        if args.voxel_size:
            size = [float(v) for v in args.voxel_size.split(",")]
            for bank in banks:
                bank["size"] = size

        monitors = collect_monitors(handle, entry, inst)
        source_z = source_position(handle, inst)
        print("  source at z = %.4f m, %d monitor(s)" % (source_z, len(monitors)))

        if not args.no_event_check:
            check_event_ids(entry, banks)

        all_ids = np.concatenate([b["ids"] for b in banks])
        if np.unique(all_ids).size != all_ids.size:
            raise SystemExit("Detector IDs are not unique across banks - refusing to "
                             "write an IDF that would silently mix pixels")

    write_idf(args.out, banks, monitors, source_z, inst_name=args.name,
              flat=args.flat_idf)

    if args.with_geom_xml:
        write_geom_xml(banks, monitors, source_z, inst_name=args.name)

    if args.embed:
        embed_idf(args.nxs, args.out, args.embed)
        print("\nNext step in Mantid:\n"
              "    ws = LoadEventNexus(Filename='%s')\n"
              "    # then right-click the workspace -> Show Instrument" % args.embed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
