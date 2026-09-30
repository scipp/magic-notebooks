#!/usr/bin/env python3
"""
make_mantid_file.py - turn an ESS file-writer NeXus file into one that Mantid's
LoadEventNexus can actually read, and put the IDF inside it.

Why: mccode.nxs contains several things the NeXus reader inside Mantid does not
cope with. The dump of the file shows

  * ~50 datasets are VARIABLE LENGTH strings (h5py dtype=object): /entry/start_time,
    /entry/title, /entry/instrument/name, every depends_on, the whole /entry/sample
    group ... The NeXus C API Mantid uses expects fixed length NX_CHAR and is the
    most likely cause of the hard crash.
  * /entry/instrument/beam_monitor_1/beam_monitor_1_events/event_id is float64;
    event IDs must be integers.
  * every event_time_zero has a 'start' attribute but no 'offset', which is the
    one Mantid reads for the absolute pulse time.
  * the NXoff_geometry meshes are huge (2.9 M faces for bank A). They are not
    needed once the IDF is embedded, and parsing them is slow.

What this script does (on a COPY - the original is never modified):

    1. rewrites every variable length string as a fixed length one, attributes kept
    2. converts non-integer event_id datasets to uint32
    3. adds the missing 'offset' attribute to event_time_zero
    4. optionally deletes the NXoff_geometry meshes          (--drop-mesh)
    5. optionally deletes NXlog groups with an empty value   (--drop-empty-logs)
    6. hard-links the nested NXevent_data groups at the top level of /entry, which
       is the only place LoadEventNexus looks for banks (--no-link-events disables)
    7. embeds the IDF as /entry/instrument/instrument_xml    (--idf FILE)

Usage
    python3 make_mantid_file.py mccode.nxs mccode_mantid.nxs --idf MAGiC_Definition.xml
    python3 make_mantid_file.py mccode.nxs mccode_mantid.nxs --idf MAGiC_Definition.xml \\
            --drop-mesh --drop-empty-logs
    python3 make_mantid_file.py --check mccode_mantid.nxs      # verify the result

Note: HDF5 does not release space when datasets are deleted, so --drop-mesh does
not make the file smaller. Run `h5repack in.nxs out.nxs` afterwards if you care.
"""
import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import MAGIC as M                                              # noqa: E402


def fixed(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return np.bytes_(value)


def walk_all(group, prefix=""):
    """(path, object) for every item in the file, depth first."""
    for key in sorted(group.keys()):
        obj = group[key]
        path = prefix + "/" + key
        yield path, obj
        if M.is_group(obj):
            for item in walk_all(obj, path):
                yield item


def rewrite_dataset(parent, key, data, attrs):
    del parent[key]
    ds = parent.create_dataset(key, data=data)
    for name, value in attrs.items():
        ds.attrs[name] = value
    return ds


def link_event_groups(handle, entry_name="entry"):
    """Hard-link nested NXevent_data groups at the top level of the entry.

    CHANGED: ESS files keep the events inside the detector group
    (/entry/instrument/magic_detector_a/magic_detector_a_event_data), while
    LoadEventNexus looks for them directly under the entry, the way SNS files are
    laid out. Without this the instrument loads and the workspace stays EMPTY -
    622592 spectra and 0 events, with no error message.
    A hard link copies no data and leaves the original group untouched.
    """
    entry = handle[entry_name]
    linked = 0
    for path, group in list(walk_all(handle[entry_name], "/" + entry_name)):
        if not M.is_group(group) or M.nx_class(group) != "NXevent_data":
            continue
        if path.count("/") - 2 == 0:
            continue                                   # already at the top level
        name = path.rsplit("/", 1)[-1]
        if name in entry:
            continue
        entry[name] = handle[path]
        linked += 1
    return linked


def sanitize(handle, drop_mesh=False, drop_empty_logs=False, verbose=False,
             link_events=True):
    stats = {"strings": 0, "event_id": 0, "offset": 0, "mesh": 0, "logs": 0,
             "links": 0}
    todo_strings, todo_ids, todo_offset, todo_mesh, todo_logs = [], [], [], [], []

    for path, obj in walk_all(handle):
        if M.is_group(obj):
            cls = M.nx_class(obj)
            if drop_mesh and cls == "NXoff_geometry":
                todo_mesh.append(path)
            elif drop_empty_logs and cls in ("NXlog",) and "value" in obj \
                    and getattr(obj["value"], "shape", (1,)) == (0,):
                todo_logs.append(path)
            continue
        dtype = getattr(obj, "dtype", None)
        if dtype is not None and dtype.kind in ("O", "U"):
            todo_strings.append(path)
        base = path.rsplit("/", 1)[-1]
        if base == "event_id" and dtype is not None and dtype.kind not in ("u", "i"):
            todo_ids.append(path)
        if base == "event_time_zero" and "offset" not in obj.attrs:
            todo_offset.append(path)

    for path in todo_strings:
        obj = handle[path]
        parent, key = handle[path.rsplit("/", 1)[0] or "/"], path.rsplit("/", 1)[-1]
        value = M.decode(obj[()])
        attrs = dict(obj.attrs)
        if isinstance(value, np.ndarray):          # array of strings
            data = np.array([fixed(M.decode(v)) for v in value.ravel()])
        else:
            data = fixed(value if isinstance(value, (str, bytes)) else str(value))
        rewrite_dataset(parent, key, data, attrs)
        stats["strings"] += 1
        if verbose:
            print("    string  %s" % path)

    for path in todo_ids:
        obj = handle[path]
        parent, key = handle[path.rsplit("/", 1)[0]], path.rsplit("/", 1)[-1]
        values = np.asarray(obj[()])
        rounded = np.rint(values)
        if np.abs(values - rounded).max() > 1.e-6:
            print("    WARNING: %s has non-integer values, rounding anyway" % path)
        rewrite_dataset(parent, key, rounded.astype(np.uint32), dict(obj.attrs))
        stats["event_id"] += 1
        if verbose:
            print("    event_id %s -> uint32" % path)

    for path in todo_offset:
        obj = handle[path]
        start = obj.attrs.get("start", b"1970-01-01T00:00:00Z")
        obj.attrs["offset"] = fixed(M.decode(start))
        stats["offset"] += 1

    if link_events:
        stats["links"] = link_event_groups(handle)
        if verbose and stats["links"]:
            print("    linked %d event group(s) at the top level" % stats["links"])

    for path in todo_mesh + todo_logs:
        parent = handle[path.rsplit("/", 1)[0]]
        del parent[path.rsplit("/", 1)[-1]]
        stats["mesh" if path in todo_mesh else "logs"] += 1
        if verbose:
            print("    deleted %s" % path)

    return stats


def check(path):
    import h5py
    problems = 0
    with h5py.File(path, "r") as handle:
        for item_path, obj in walk_all(handle):
            if M.is_group(obj):
                continue
            dtype = getattr(obj, "dtype", None)
            base = item_path.rsplit("/", 1)[-1]
            if dtype is not None and dtype.kind in ("O", "U"):
                print("  still a variable length string: %s" % item_path)
                problems += 1
            if base == "event_id" and dtype is not None and dtype.kind not in ("u", "i"):
                print("  event_id is still %s: %s" % (dtype, item_path))
                problems += 1
            if base == "event_time_zero" and "offset" not in obj.attrs:
                print("  event_time_zero without offset: %s" % item_path)
                problems += 1
        entry = M.find_entry(handle)
        top_level = [k for k in entry.keys()
                     if M.is_group(entry[k]) and M.nx_class(entry[k]) == "NXevent_data"]
        print("  %d NXevent_data group(s) at the top level of the entry%s"
              % (len(top_level), "" if top_level else "  <-- Mantid will load 0 events"))
        inst = M.find_instrument(entry)
        if "instrument_xml" in inst:
            size = inst["instrument_xml/data"].nbytes / 1024.
            print("  instrument_xml present (%.0f kB)" % size)
        else:
            print("  no instrument_xml - Mantid will fall back to the NeXus geometry")
    print("  %d problem(s) left" % problems)
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="input NeXus file (never modified)")
    parser.add_argument("target", nargs="?", help="output NeXus file")
    parser.add_argument("--idf", default=None, help="IDF to embed")
    parser.add_argument("--drop-mesh", action="store_true",
                        help="delete the NXoff_geometry meshes (not needed with an IDF)")
    parser.add_argument("--drop-empty-logs", action="store_true",
                        help="delete NXlog groups whose value array is empty")
    parser.add_argument("--no-link-events", action="store_true",
                        help="do not hard-link the nested NXevent_data groups at the "
                             "top level (Mantid then loads 0 events)")
    parser.add_argument("--check", action="store_true",
                        help="only verify an already converted file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    try:
        import h5py
    except ImportError:
        raise SystemExit("h5py is required (conda run -n py python make_mantid_file.py ...)")

    if args.check:
        return 1 if check(args.source) else 0
    if not args.target:
        raise SystemExit("Give an output file name (or use --check)")
    if os.path.abspath(args.source) == os.path.abspath(args.target):
        raise SystemExit("Refusing to overwrite the original file")

    print("Copying %s -> %s (%.0f MB) ..."
          % (args.source, args.target, os.path.getsize(args.source) / 1.e6))
    shutil.copyfile(args.source, args.target)

    print("Sanitizing ...")
    with h5py.File(args.target, "r+") as handle:
        stats = sanitize(handle, drop_mesh=args.drop_mesh,
                         drop_empty_logs=args.drop_empty_logs, verbose=args.verbose,
                         link_events=not args.no_link_events)
    print("  fixed-length strings : %d" % stats["strings"])
    print("  event_id -> uint32   : %d" % stats["event_id"])
    print("  offset attributes    : %d" % stats["offset"])
    print("  meshes deleted       : %d" % stats["mesh"])
    print("  empty logs deleted   : %d" % stats["logs"])
    print("  event groups linked  : %d  (LoadEventNexus finds banks only at the "
          "top level of /entry)" % stats["links"])

    if args.idf:
        print("Embedding %s ..." % args.idf)
        M.embed_idf(args.target, args.idf)

    print("Checking the result ...")
    check(args.target)
    print("\nNow in Mantid:\n    ws = LoadEventNexus(Filename='%s')" % args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
