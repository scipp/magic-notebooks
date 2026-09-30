"""MAGiC instrument definition built from voxelization.py (plan section 100).

    python3 make_idf_voxelization.py --mcstas fe4o5_1.h5 \
        --template coda_magic_999999_00016485.hdf --out MAGiC_Definition_vox.xml

voxelization.py is the procedure that gives a detected neutron its detector ID
(read_h5.py -> batch_convert.py write exactly those IDs when --geometry is NOT
given), so the positions Mantid uses must come from the same model - not from
the detector_faces mesh of the CODA template, which is outdated (Iurii,
28 Sep 2026) and numbers the voxels differently (n_vs runs the other way).

* Both banks at their REFERENCE angle, gamma_d = 0, the convention of
  MAGiC_Definition_zero.xml: the run supplies the bank angle
  (entry/instrument/detector_a_rotation), magic_workflow.py turns the bank.
* The segment tilt omega_vs of each bank is what McStas simulated:
  A_casette_omega / B_casette_omega from the McStas parameters (--mcstas), or
  --omega-a / --omega-b.
* --pairs plane (default): voxelization.calc_xyz_by_id - both voxels of a pair
  at the plane they share, as the model defines them.  --pairs centre: each
  voxel at the middle of its own half of the segment (+-delta_gamma/4), which
  is where the absorbed McStas neutrons of that ID lie on average if the
  simulated absorber fills the segment rather than sitting on the plane.
* Monitors from --template (the CODA file), the source at --source-z, default
  -154.4988 m: the real moderator-to-sample distance in MAGiC_instr.instr (not
  the 159.397 of its sourceMantid arm, section 110).
"""
import argparse
import sys

import numpy as np

import MAGIC
import voxelization


def bank_a(omega_deg):
    return voxelization.DetectorA(omega_vs=np.radians(omega_deg), gamma_d=0.0)


def bank_b(omega_deg):
    # the parameters voxelization_of_mcstas_events_for_detector_b uses
    return voxelization.DetectorA(
        N_vs=16*8, N_a=16, N_c=32, r_d=1.000, r_vs=0.530,
        delta_gamma_vs=np.radians(-0.94),
        a_t=0.051016, b_t=0.077887, a_b=-0.051016, b_b=-0.077887,
        omega_vs=np.radians(omega_deg), gamma_d=0.0, n_id_0=491521)


def positions(det, pairs):
    count = 2*det.N_vs*det.N_a*det.N_c
    ids = np.arange(det.n_id_0, det.n_id_0 + count, dtype=np.int64)
    if pairs == "plane":
        pos = det.calc_xyz_by_id(ids).T
    else:
        n_vs, n_a, n_c = det._calc_n_vsac_by_id(ids)
        x, y = det._calc_xy_vs_by_n_ac(n_c, n_a)
        side = np.where(n_vs % 2 == 0, -1.0, 1.0)
        gamma = (det.gamma_d + np.floor_divide(n_vs, 2)*det.delta_gamma_vs
                 + side*0.25*det.delta_gamma_vs)
        pos = np.stack((det.r_d*np.sin(gamma) + x*np.sin(gamma + det.omega_vs),
                        y,
                        det.r_d*np.cos(gamma) + x*np.cos(gamma + det.omega_vs)),
                       axis=1)
    back = det.calc_id_by_xyz(pos.T.copy())
    exact = float(np.mean(back == ids))
    within = float(np.mean(np.abs(back - ids) <= 1))
    print("    ids %d..%d: position -> id gives the same id for %.1f %%, the "
          "same or its pair partner for %.1f %%%s"
          % (ids[0], ids[-1], 100*exact, 100*within,
             "" if pairs == "centre" else
             " (plane: a pair shares one position, so ~50 % exact is right)"))
    depth = det.r_vs/det.N_c
    xc = (np.arange(det.N_c) + 0.5)/det.N_c
    height = float(np.mean(((det.a_t + (det.b_t - det.a_t)*xc)
                            - (det.a_b + (det.b_b - det.a_b)*xc))/det.N_a))
    width = float(abs(det.delta_gamma_vs)/2*(det.r_d + 0.5*det.r_vs))
    return ids, pos, np.array([width, abs(height), depth])


def casette_omegas(path):
    import h5py
    found = {}
    with h5py.File(path, "r") as handle:
        for key, value in handle["entry1/simulation/Param"].items():
            if key in ("A_casette_omega", "B_casette_omega"):
                found[key[0]] = float(value[()][0].decode("ascii"))
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="MAGiC_Definition_vox.xml")
    parser.add_argument("--mcstas", help="a McStas .h5, for the casette omegas")
    parser.add_argument("--omega-a", type=float)
    parser.add_argument("--omega-b", type=float)
    parser.add_argument("--pairs", choices=("plane", "centre"), default="plane")
    parser.add_argument("--template", help="CODA file, for the monitors")
    parser.add_argument("--source-z", type=float, default=-154.4988)  # section 110: the McStas moderator-to-sample distance
    parser.add_argument("--no-bank-b", action="store_true")
    args = parser.parse_args(argv)

    omega = {}
    if args.mcstas:
        omega.update(casette_omegas(args.mcstas))
        print("from %s: %s" % (args.mcstas, ", ".join(
            "%s_casette_omega %.3f deg" % kv for kv in sorted(omega.items()))))
    if args.omega_a is not None:
        omega["A"] = args.omega_a
    if args.omega_b is not None:
        omega["B"] = args.omega_b
    if "A" not in omega:
        sys.exit("no omega for bank A: give --mcstas or --omega-a")

    banks = []
    for letter, maker in (("A", bank_a), ("B", bank_b)):
        if letter == "B" and (args.no_bank_b or "B" not in omega):
            print("bank B left out%s" % ("" if args.no_bank_b else
                  " - no B_casette_omega (give --mcstas with bank B, or --omega-b)"))
            continue
        print("bank %s, omega_vs %.3f deg, pairs at the %s, reference angle 0"
              % (letter, omega[letter], args.pairs))
        ids, pos, size = positions(maker(omega[letter]), args.pairs)
        banks.append({"name": "magic_detector_%s" % letter.lower(), "ids": ids,
                      "pos": pos, "size": size, "source": "voxelization.py"})
        MAGIC.describe_bank(banks[-1])

    monitors = []
    if args.template:
        import h5py
        with h5py.File(args.template, "r") as handle:
            entry = MAGIC.find_entry(handle)
            inst = MAGIC.find_instrument(entry)
            monitors = MAGIC.collect_monitors(handle, entry, inst)
    MAGIC.write_idf(args.out, banks, monitors, args.source_z)
    print("written: %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
