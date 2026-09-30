# MAGiC single-crystal reduction for McStas simulations

Two steps turn a single-crystal simulation of MAGiC (ESS) made with McStas into
integrated intensities:

```
McStas (MAGiC_instr.instr)  ->  *.h5
batch_convert.py            ->  *.nxs   ESS NeXus event files, as the instrument will write them
magic_workflow.py (Mantid)  ->  *.csv, *.int   h k l, I, sigma, ...
```

Nothing in the chain is specific to one crystal: the sample enters only
through the cell given to the reduction.

---

## 1. Contents

```
batch_convert.py          step 1: McStas .h5 -> NeXus (scipp environment)
magic_workflow.py         step 2: NeXus -> integrated intensities (Mantid)
README_magic_workflow.md  this guide
dependencies/             used by batch_convert.py, not run directly
    read_h5.py                     reads the McStas detector events
    voxelization.py                the detector voxel model: assigns the detector IDs
    make_idf_voxelization.py       builds the Mantid instrument definition from voxelization.py
    MAGIC.py, make_mantid_file.py  write the instrument definition, make the file loadable by Mantid
```

Keep the folder structure as it is.

Not included, to be given on the command line: **a CODA file** of MAGiC
(`--template`), e.g. `coda_magic_999999_00016485.hdf`. The converter copies its
NeXus layout (groups, monitors, metadata) and replaces the events, so the
output looks like a file the instrument writes. The chain was validated with
`coda_magic_999999_00016485.hdf`; giving a newer CODA file is how the same
chain is tested against new versions of the instrument's file layout.

---

## 2. Software

* **Step 1:** Python 3 with `scipp`, `h5py`, `numpy`, `scipy` (no Mantid),
  e.g. `conda create -n scipp -c conda-forge scipp h5py scipy`.
* **Step 2:** Mantid 6.16 (tested with 6.16.1), run with Mantid's own Python
  (the Mantid conda environment, `mantidpython`, or the Workbench script
  editor); numpy, scipy and h5py come with it.

**Do not put a `MAGiC_Parameters.xml` into Mantid's instrument directory**
(`~/.mantid/instrument`). An old version carries t0 = 3000 µs and makes Mantid
shift the time of flight of every file after the first one. t0 is given on the
command line (`--t0`).

---

## 3. What the simulation has to provide

* **Instrument:** `MAGiC_instr.instr` (McStas 3.7.x), thermal spectrum
  (`isCold=0`). The calibration used below (t0 = 1715 µs, flight path from the
  McStas geometry, band 0.65–2.25 Å) was made for this setting.
* **One file per crystal orientation.** The sample component must be called
  `sampleMantid`, as in the instrument file: the converter reads the crystal
  rotation from its rotation matrix. Only rotations about the vertical are
  supported; a chi/phi rotation is flagged in the converter's table. The bank
  angle `da_gamma` can be anything; it is read from each file.
* **A vanadium run** with the same instrument settings, for the normalisation:
  `NCrystal_sample(cfg = "V_sg229.ncmat;temp=300", radius = 0.001*sample_radius)`
  in place of the crystal. One vanadium file serves every run made with the
  same settings.
* **Crystal lattice in `Single_crystal` given as vectors** (`ax, ay, az, bx,
  ... cz`), or a McStas version with the patch for the "lengths and angles"
  mode. In McStas 3.7.x that mode (cell given as `aa, bb, cc`, or read from
  the header of the reflection file, e.g. written by cif2hkl) swaps a and b:
  every |F(h,k,l)|² ends up at the position of (k,h,l), and the reduced
  intensities then belong to (k,h,l) of the reflection file. A cubic cell, or
  a = b with α = β, is not affected.

---

## 4. Step 1 – convert (scipp environment)

```bash
python3 batch_convert.py --mcstas 'mysample_*.h5' --template CODA_FILE.hdf --tof-window 22,92
python3 batch_convert.py --mcstas vanadium.h5 --template CODA_FILE.hdf --tof-window 22,92
```

`mysample_1.h5` becomes `mysample_1.nxs` (`--outdir` puts them elsewhere).
For every batch the converter

* takes the NeXus layout of the CODA file and fills it with the simulated
  events;
* builds the Mantid instrument definition from `voxelization.py` and the
  segment tilt of the McStas file, writes it as `MAGiC_Definition_vox.xml` and
  embeds it in every output file, so the reduction needs no separate geometry
  file;
* keeps the McStas events inside the voxel model (this assigns the detector
  IDs) and draws 10 M events per bank from the McStas weights over 1100 pulses
  at 14 Hz (`--events-a`, `--pulses`);
* reads the crystal rotation and the bank angles from the McStas file and
  writes them where the instrument will write them;
* folds the time of flight into the 71.43 ms frame (`--tof-window 22,92`): the
  slow neutrons of the previous pulse appear at the start of the frame, as on
  the real instrument; the reduction unfolds them (`--unwrap auto`).

It prints a table of the files and their rotation angles; check that they are
the ones you simulated. A few minutes per file. `--dry-run` only prints the
table; `--help` lists the options.

---

## 5. Step 2 – reduce (Mantid environment)

```bash
python3 magic_workflow.py mysample_1.nxs mysample_2.nxs mysample_3.nxs \
    --cell A,B,C[,ALPHA,BETA,GAMMA] --centring LETTER \
    --t0 1715 --unwrap auto --lambda 0.65,2.25 --two-theta-min 20 \
    --norm vanadium.nxs --out mysample 2>&1 | tee mysample.log
```

| Option | Meaning |
|---|---|
| `--cell A,B,C[,AL,BE,GA]` | the cell given to McStas (Å, degrees). Three numbers: angles 90°; one number: cubic. Only each run's orientation is then searched and the cell refined; its equalities (a = b, 90°, 120°) are kept. Without `--cell` a free lattice search is tried, which is not reliable on these data. |
| `--centring P/A/B/C/I/F/Robv/Rrev` or `--space-group NAME` | which reflections can exist (of a space group only the centring letter is used) |
| `--t0 1715` | emission-time offset, µs (calibrated on simulated Ge and diamond powders) |
| `--unwrap auto` | moves the previous pulse's neutrons back to t + 71.43 ms |
| `--lambda 0.65,2.25` | wavelength band; below 0.65 Å the thermal spectrum is a thin tail |
| `--two-theta-min 20` | masks the direct-beam edge of the bank for the peak search |
| `--norm vanadium.nxs` | normalisation from the vanadium: spectrum × transmission of the detector depth × voxel efficiency |
| `--out STEM` | stem of the output files |

`--steps load,propose` stops after loading and prints what the data show
(band, angular range, peaks) – a cheap first look. `--help` lists every
option; the most useful others are `--d-range LO,HI` (limit the predicted
reflections) and `--peak-sigmas R,T` (fix the integration ellipsoid by hand).

The log explains each step as it runs: load and unfold the events → peak
search in Q → resolution function from the strongest peaks → indexing with the
given cell, one orientation per run → UB refinement per run → normalisation →
every reflection the lattice predicts in the band and on the bank is
integrated in its own ellipsoid, whose size along and across Q is measured
(the smallest size holding ≥ 99 % of the strong peaks' intensity in every |Q|
band) → output. Three runs take about 3 minutes and 3–4 GB of memory; Mantid
warnings beyond the first 20 go to `STEM_messages.log`.

### Output

| File | Content |
|---|---|
| `STEM_<run>.csv` | h, k, l, intensity, sigma, d, λ, 2θ, coverage, kept, detector id, raw counts, normalisation factors |
| `STEM_all.csv` | all runs in one table, with a run column |
| `STEM_<run>.int` | Fullprof format, for refinement programs |
| `STEM_<run>_shape.npz` | net intensity for every ellipsoid size (diagnostics) |
| `STEM.log`, `STEM_messages.log` | the log and the suppressed Mantid messages |

Intensities are corrected for the Lorentz factor (sin²θ/λ⁴) and normalised by
the vanadium; the absolute scale is arbitrary. `kept = 0` marks reflections cut
by the bank edge or the band (coverage < 0.6); leave them out.

---

## 6. Troubleshooting

* **Few or no peaks indexed:** check the cell and the centring, run
  `--steps load,propose`, and check the angle table of the converter.
* **Intensities do not match the reflection file, many strong ones where F² ≈ 0:**
  most likely the a/b swap of section 3; compare with the file's (k,h,l).
* **Time of flight shifted by 3 ms between files:** a `MAGiC_Parameters.xml`
  in Mantid's instrument directory (section 2).

---

## 7. Validation: simulated Fe4O5

Cmcm, a = 2.8906, b = 9.8024, c = 12.5804 Å, three crystal orientations,
McStas 3.7.14 (cell given as lengths and angles, so compared as (k,h,l)),
reduced with `--cell 2.89060,9.80240,12.58040 --centring C` and the options of
section 5 (Mantid 6.16.1, 29 Sep 2026):

* indexed peaks 86 / 75 / 39; ellipsoid 3.5 σ along Q × 5.5 σ across;
* against the |F|² of the reflection file, reflections with I/σ > 3, one scale
  per run: correlation 0.96, R(F²) 0.19; symmetry-equivalent reflections agree
  to R_int 0.016;
* merged to unique reflections: R(F²) 0.14 (all d), 0.12 (d ≥ 0.7 Å),
  correlation of the logarithms 0.99.

---

## 8. Known limitations

* Reflections with d < 0.7 Å (|Q| 9–11 Å⁻¹, λ 0.66–0.75 Å) come out 20–40 %
  too strong; not understood yet. `--d-range 0.7,20` leaves them out.
* Separate runs come out on scales up to ~20 % apart; merging runs needs a
  common-scale refinement afterwards.
* The refined cell is ~0.8 % too large (weak high-|Q| peaks pull it); the
  given cell is the better value.
* The normalisation uses the efficiency of the voxel a peak is assigned to (the
  front layer), while its counts come from all 32 depth layers.
* Calibrated for the thermal spectrum only. t0 and the flight path cannot be
  separated with any powder in this band; the flight path is taken from the
  McStas geometry.
* Only bank A is reduced (bank B is masked).
