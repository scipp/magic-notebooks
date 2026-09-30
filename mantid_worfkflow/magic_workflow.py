"""MAGiC: from bare NeXus files to a file of integrated reflections.

Nothing is assumed about the sample.  The crystal system is the only thing the
operator supplies; the cell, the orientation of each run, the usable part of the
spectrum and the resolution function are all measured from the data.

    python3 magic_workflow.py run1.nxs run2.nxs run3.nxs --system orthorhombic
    python3 magic_workflow.py *.nxs --cell A,B,C --norm vanadium.nxs --out mysample
    python3 magic_workflow.py run1.nxs --steps load,propose      look first
    python3 magic_workflow.py --help

Several runs at once, one normalisation file for all of them.  What belongs to a
run is measured per run - the frame, the peak list, the orientation.  What
belongs to the instrument or the crystal is measured once from everything
together: the resolution function, because it is the instrument, and the cell,
because it is the crystal, with one orientation per run.

Run it inside MantidWorkbench, or with the python that ships with it.

Workspaces are LEFT IN PLACE so that every step can be looked at afterwards.
The exception is the MD workspace of each run, which for a full MAGiC bank runs
to several GB and will hang the interface; --keep-md keeps them.

What this workflow can and cannot know
--------------------------------------
It can find the frame, the usable wavelength band, the peaks, the resolution
function, the cell and the orientation of every run.

It cannot know t0 - the time the neutron left the moderator - because t0 and the
flight path L enter the time of flight only as t = t0 + (L/3.956e-3) lambda, and
no single measurement separates them; a powder does not either, because the d it
reaches goes as 1/sin(theta) and moves with the same lever.  A wrong t0 leaves
the SHAPE of the cell right and its SCALE wrong, with the error going as
1/lambda.  So the cell that comes out of here is a shape that belongs to the
crystal and a scale that belongs to whatever calibration set --t0.
"""

import itertools
import math
import os
import sys
import time

import numpy as np

from mantid.simpleapi import (
    LoadEventNexus, LoadInstrument, CropWorkspace, ChangeBinOffset,
    ConvertUnits, Rebin, SumSpectra, SetGoniometer, MaskDetectors,
    PreprocessDetectorsToMD, RotateInstrumentComponent, DeleteWorkspace, mtd)
# Everything else - ConvertToMD, FindPeaksMD, CentroidPeaksMD, BinMD,
# IntegratePeaksMD, PredictPeaks, FilterPeaks, CombinePeaksWorkspaces, SetUB,
# CloneWorkspace, Integration, SaveReflections - goes through call() below,
# which checks property names against the installed Mantid instead of trusting
# them.
from mantid.api import AlgorithmManager
from mantid.kernel import ConfigService

# ------------------------------------------------------------- what to run --
RUNS = []                   # NeXus files, from the command line
NORMALISATION = None        # one more NeXus file, vanadium; --norm
SYSTEM = "orthorhombic"     # --system: triclinic monoclinic orthorhombic
                            # tetragonal rhombohedral hexagonal cubic.  Used
                            # ONLY when no cell is given; a cell says the same
                            # thing more precisely, and the two would conflict.
SYSTEM_GIVEN = False        # whether --system was actually passed
CELL = None                 # --cell a,b,c[,alpha,beta,gamma]; then only the
                            # orientation of each run is unknown
REFINE = "cell"             # --refine none | orientation | cell | a,b,c,...
REFINE_NAMES = None         # the named parameters, when --refine listed them
CENTRING = "P"              # --centring P A B C I F Robv Rrev
D_RANGE = None              # --d-range lo,hi; else from the bank and the band
LORENTZ = True              # --no-lorentz turns it off
IDF = None                  # --idf, replaces the geometry in the files
BANK_GAMMA = None           # --gamma, deg: the angle the movable bank STOOD AT
                            # in this run
IDF_GAMMA = 0.0             # --idf-gamma, deg: the angle the geometry file
                            # already carries.  The rotation applied is
                            # -(BANK_GAMMA - IDF_GAMMA), so when the file was
                            # built from this very run the two are equal and
                            # nothing is turned.
                            #
                            # A geometry file ought to carry the bank at a
                            # reference angle and let the run say where it
                            # actually stood, because the bank moves.  The file
                            # used here does not: its voxel positions are
                            # absolute, with whatever angle its source run used
                            # baked in, so it is strictly valid for that one
                            # angle.  Hence --idf-gamma, which is a workaround
                            # for a geometry file, not a physical quantity.
                            # The sign is the one the working reduction uses.
BANK_GAMMA_B = None         # --gamma-b, the same for bank B
IDF_GAMMA_B = None           # --idf-gamma-b; None means bank B is left alone,
                            # because a geometry file that does not say what
                            # angle it was built at cannot be corrected
BANK_COMPONENT = None       # --bank-component NAME; found in the file if absent
BANK_COMPONENTS = {"a": ("magic_detector_a", "bank_a", "detector_a", "bank1"),
                   "b": ("magic_detector_b", "bank_b", "detector_b", "bank2")}
T0 = None                   # --t0, microseconds.  Left out, it is read from
                            # the instrument's 't0' parameter (MAGiC_Parameters
                            # .xml carries 3000, the ESS long-pulse correction
                            # measured on Ge); 0 only if neither exists
T0_SOURCE = ""              # where T0 came from, for the printed text
BANK = "a"                  # --bank a|b: the ONE bank whose events go into
                            # Q space.  A and B are different detectors - B
                            # will sit behind a polarisation analyser - so
                            # their data are never mixed; the other bank is
                            # masked from 'prepare' on
AUTO = True                 # --ask makes it stop at every proposal
KEEP_MD = False             # --keep-md
OUTPUT = "workflow"         # --out, the stem for the files written

TWO_THETA_MIN = None        # --two-theta-min, else proposed from the data
LAMBDA_BAND = None          # --lambda lo,hi, else proposed from the data
PEAK_SIGMAS = 3.0           # --peak-sigmas X or R,T, the integration ellipsoid in
PEAK_SIGMAS_R = 3.0         # along Q  } section 117: the two semi-axes are
PEAK_SIGMAS_T = 3.0         # across Q } measured separately
                            # sigmas of the ONE resolution function; with
                            # PEAK_SIGMAS_AUTO it is measured (section 115)
PEAK_SIGMAS_AUTO = True     # --peak-sigmas auto (default): the smallest n at
                            # which the strong peaks stop growing
SATURATION_N = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
SATURATION_KEEP = 0.99      # section 118: the ellipsoid holds >= 99 % of the
                            # plateau (the net at the largest n) in EVERY band
                            # of |Q| - a median over all |Q| let the high-|Q|
                            # peaks, wider across Q than the fitted function,
                            # lose 3-5 %
SATURATION_BAND_MIN = 8     # strong reflections per |Q| band
SATURATION_BINS = (71, 61, 61)
RADIAL_CONSTANT_MAX = 0.06  # A-1: a larger fitted radial constant is not
                            # trusted (section 86) - proportional instead
CENTROID_RADIUS = 0.15      # --centroid-radius A-1 for centroiding the predicted
                            # reflections (0.15 as sx_magic, section 113);
                            # 0 = 2 sigma_r at the mean |Q|
UNWRAP = None               # --unwrap auto|MS: neutrons of the PREVIOUS pulse
                            # (lambda above one frame's band) are recorded at
                            # t - 71.43 ms, early in the frame.  Events below
                            # the split go to t + 71.43 ms before anything
                            # else.  'auto' puts the split in the middle of the
                            # empty stretch of the recorded frame; a number is
                            # the split in ms.  Plan section 98.
MAX_CELL_DRIFT = 0.03       # a per-run cell further than this is refused
UB_PER_RUN = True           # --ub-per-run / --common-ub: as sx_magic, refine UB of each run on
                            # its centroided predictions, predict again and
                            # centroid again before integrating
PEAK_DISTANCE = 0.25        # --peak-distance, A-1: FindPeaksMD merges peaks
                            # closer than this.  Mantid's default is 0.1, and a
                            # MAGiC peak is a streak about 0.3 A-1 long along
                            # Q at |Q| = 5 (sigma_r = 0.017|Q|), so at 0.1 one
                            # reflection comes out as several peaks strung along
                            # its radial line: four times the peaks sx_magic
                            # finds at the same density, and nothing indexes.
                            # 0.25 is what sx_magic has always used.
BACK_SIGMAS = (3.0, 4.5)    # the background shell, in sigmas
RESOLUTION_PEAKS = 40       # strong peaks per run for the resolution function
INDEX_PEAKS = 45            # brightest peaks per run used to find the lattice
INDEX_TOLERANCE = 2.5       # kept for the printed text; the windows that
                            # decide are INDEX_SIGMA_R and INDEX_SIGMA_T
MIN_COVERAGE = 0.6          # a reflection whose ellipsoid is cut below this
MIN_PEAKS, MAX_PEAKS = 150, 800
DENSITY_LADDER = (1000, 1400, 2000, 2800, 4000, 6000, 10000)   # as sx_magic
NORM_TARGET = 0.05          # wanted error per cell of the normalisation table
NORM_LAMBDA_BINS = 40
NORM_MODEL = "depth"        # --norm-model depth|groups (section 116): depth is
                            # sx_magic's Phi(lambda) x exp(-k(depth) lambda) x
                            # epsilon(voxel); groups is the empirical table of
                            # section 82 over (2theta x L2) groups
BANK_A_SHAPE = (120, 128, 32)   # voxels along a row (pairs), rows, depth levels
BIG_MB = 2000               # a workspace this size is dropped unless kept
MAX_MASK = 0.25             # an angular mask may not eat more than this
MAX_NODES_PER_PEAK = 50     # a lattice denser than this is refused

PREFIX = "wf"
_FRAME_US = 1.0e6/14.0      # the ESS repetition rate
_CONVERT = 3.9560e-3        # h/m, in A m per us

for _quiet in (lambda: ConfigService.setLogLevel(4),
               lambda: ConfigService.Instance().setLogLevel(4)):
    try:
        _quiet()
        break
    except Exception:
        continue

SHARED = {}                 # the instrument and the crystal
RUN = {}                    # tag -> everything about that run
CHAIN = [False]             # True while main() is walking the whole list


def need(step, what):
    """Run a missing earlier step, but only when this one was asked for alone.

    During a full run every step is going to happen anyway, so calling backwards
    just prints the same thing several times over - which is exactly what it did
    when a step never got what it needed.
    """
    if CHAIN[0]:
        say("   nothing to work with: %s.  Earlier steps did not produce it."
            % what)
        return False
    step()
    return True


# ------------------------------------------------------- noise control --
# Mantid and numpy can print thousands of near-identical Warning and Error
# lines - one per spectrum, one per peak - and bury what the workflow itself
# says.  Every such line is COUNTED; the first MAX_WARNINGS warnings and the
# first MAX_ERRORS errors are shown, the rest are suppressed with one notice,
# and ALL of them go to <out>_messages.log so nothing is lost.  The workflow's
# own text (say) is never counted or suppressed.
#
# From a terminal (mantidpython magic_workflow.py ...) the file descriptors 1
# and 2 are captured, so this covers Mantid's C++ log as well.  Inside
# Workbench only the Python streams can be caught: Mantid's own Messages panel
# is outside a script's reach, and there the log level is the only control.
MAX_WARNINGS = 20           # --max-warnings N, -1 for no limit
MAX_ERRORS = 100            # --max-errors N,   -1 for no limit

import re as _re
import threading as _threading

_MARK = "\x01"              # prefix of a line that comes from say()
_IS_ERROR = _re.compile(r"\[(error|critical|fatal)\]|\berror\b|exception",
                        _re.I)
_IS_WARNING = _re.compile(r"\[warning\]|warning", _re.I)


class _Chatter:
    """Counts, shows or suppresses Warning and Error lines."""

    def __init__(self):
        self.active = False
        self.lock = _threading.Lock()
        self.count = {"warning": 0, "error": 0}
        self.limit = {"warning": 100, "error": 100}
        self.log_path = None
        self.log = None
        self.undo = []

    # -- one line of text, from whichever stream ---------------------------
    def line(self, text, emit):
        if text.startswith(_MARK):
            emit(text[len(_MARK):])
            return
        body = text.rstrip("\n")
        kind = ("error" if _IS_ERROR.search(body) else
                "warning" if _IS_WARNING.search(body) else None)
        if kind is None:
            emit(text)
            return
        with self.lock:
            self.count[kind] += 1
            n, limit = self.count[kind], self.limit[kind]
            if self.log is None and self.log_path:
                try:
                    self.log = open(self.log_path, "w")
                except Exception:
                    self.log_path = None
            if self.log is not None:
                self.log.write(body + "\n")
        if limit < 0 or n <= limit:
            emit(text)
        elif n == limit + 1:
            emit("   [%d %ss shown; further %s messages are suppressed, all of"
                 " them go to %s]\n" % (limit, kind, kind,
                                         self.log_path or "nowhere"))

    # -- start and stop ------------------------------------------------------
    def start(self, max_warnings, max_errors, log_path):
        if self.active:
            return
        self.limit = {"warning": max_warnings, "error": max_errors}
        self.count = {"warning": 0, "error": 0}
        self.log_path = log_path
        self.active = True
        in_gui = any(m.split(".")[0] in ("workbench", "mantidqt", "ipykernel")
                     for m in sys.modules)
        if not in_gui:
            try:
                for fd in (1, 2):
                    self._capture(fd)
                return
            except Exception:
                self._release()
        self._wrap()

    def stop(self):
        if not self.active:
            return
        self._release()
        self.active = False
        with self.lock:
            if self.log is not None:
                self.log.close()
                self.log = None
        w, e = self.count["warning"], self.count["error"]
        shown = lambda n, lim: n if lim < 0 else min(n, lim)
        if w or e:
            say("messages   : %d warnings (%d shown), %d errors (%d shown)%s"
                % (w, shown(w, self.limit["warning"]),
                   e, shown(e, self.limit["error"]),
                   "; all in %s" % self.log_path if self.log_path else ""))

    def _release(self):
        while self.undo:
            try:
                self.undo.pop()()
            except Exception:
                pass

    # -- terminal: the file descriptors themselves ---------------------------
    def _capture(self, fd):
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        saved = os.dup(fd)
        read_end, write_end = os.pipe()
        os.dup2(write_end, fd)
        os.close(write_end)

        def emit(text):
            data = text.encode("utf-8", "replace")
            while data:
                data = data[os.write(saved, data):]

        def pump():
            rest = b""
            try:
                while True:
                    chunk = os.read(read_end, 65536)
                    if not chunk:
                        break
                    rest += chunk
                    *lines, rest = rest.split(b"\n")
                    for raw in lines:
                        self.line(raw.decode("utf-8", "replace") + "\n", emit)
                if rest:
                    self.line(rest.decode("utf-8", "replace"), emit)
            except Exception:
                # never let the filter block the program: pass the rest raw
                while True:
                    chunk = os.read(read_end, 65536)
                    if not chunk:
                        break
                    os.write(saved, chunk)
            finally:
                os.close(read_end)

        thread = _threading.Thread(target=pump, daemon=True)
        thread.start()

        def undo():
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except Exception:
                    pass
            os.dup2(saved, fd)          # the pipe's last writer goes: EOF
            thread.join(10.0)
            os.close(saved)
        self.undo.append(undo)

    # -- Workbench, notebooks: the Python streams only -----------------------
    def _wrap(self):
        chatter = self

        class Stream:
            def __init__(self, inner):
                self.inner, self.rest = inner, ""

            def write(self, text):
                self.rest += text
                *lines, self.rest = self.rest.split("\n")
                for one in lines:
                    chatter.line(one + "\n", self.inner.write)
                return len(text)

            def flush(self):
                try:
                    self.inner.flush()
                except Exception:
                    pass

            def __getattr__(self, name):
                return getattr(self.inner, name)

        out, err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Stream(out), Stream(err)

        def undo():
            for s in (sys.stdout, sys.stderr):
                if isinstance(s, Stream) and s.rest:
                    chatter.line(s.rest, s.inner.write)
            sys.stdout, sys.stderr = out, err
        self.undo.append(undo)


CHATTER = _Chatter()


# ---------------------------------------------------------------- helpers --
def say(text=""):
    if CHATTER.active:
        text = "\n".join(_MARK + part for part in str(text).split("\n"))
    print(text)
    try:
        sys.stdout.flush()
    except Exception:
        pass


def tick(text):
    say("   %-14s %s  %s" % ("", time.strftime("%H:%M:%S"), text))


def head(title):
    say()
    say("=" * 74)
    say("  %s" % title)
    say("=" * 74)


def call(name, **kwargs):
    """Run a Mantid algorithm, passing only properties it actually has."""
    algorithm = AlgorithmManager.create(name)
    algorithm.initialize()
    algorithm.setChild(False)
    known = {p.name for p in algorithm.getProperties()}
    unknown = sorted(set(kwargs) - known)
    for key in unknown:
        kwargs.pop(key)
    if unknown:
        say("   note: %s has no %s in this Mantid - left out"
            % (name, ", ".join(unknown)))
    for key, value in kwargs.items():
        algorithm.setProperty(key, value)
    algorithm.execute()
    out = kwargs.get("OutputWorkspace") or kwargs.get("PeaksWorkspace")
    return mtd[out] if isinstance(out, str) and out in mtd else None


def tag_of(path):
    """A short workspace-safe name for a file."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return "".join(c if c.isalnum() else "_" for c in stem)


def out_name(tag, extension):
    """The output path, without repeating the stem inside the run tag."""
    piece = tag
    stem = os.path.basename(OUTPUT)
    if piece.startswith(stem):
        # section 121: a tag equal to the stem gives STEM.ext, not STEM_STEM.ext
        piece = piece[len(stem):].strip("_")
    return "%s_%s.%s" % (OUTPUT, piece, extension) if piece else \
           "%s.%s" % (OUTPUT, extension)


def ws(tag, suffix):
    return "%s_%s_%s" % (PREFIX, tag, suffix)


def big(name):
    try:
        return mtd[name].getMemorySize()/1.0e6
    except Exception:
        return 0.0


def confirm(question):
    if AUTO:
        say("   [auto] %s -> taken" % question)
        return True
    # the question goes through say() and input() gets no prompt of its own:
    # a prompt without a newline would sit in the message filter unseen
    say("   %s  [Enter = yes, n = no]" % question)
    return not input().strip().lower().startswith("n")


def each_run():
    """The runs in the order they were given, with their state."""
    for path in RUNS:
        yield tag_of(path), path, RUN.setdefault(tag_of(path), {"path": path})


# ------------------------------------------------------------------ load --
def read_bank_angles(path):
    """The angles the movable banks stood at, from the NeXus file itself.

    These files carry entry/instrument/detector_a_rotation, so the angle is a
    measured quantity in the data and does not belong on a command line.  Mantid
    does not surface it as a run log, hence reading the file directly.
    """
    try:
        import h5py
    except ImportError:
        return {}, "h5py is not available, so the file could not be asked"
    found, notes = {}, []
    try:
        with h5py.File(path, "r") as handle:
            wanted = {}

            def visit(name, obj):
                low = name.lower()
                for letter in ("a", "b"):
                    if "detector_%s_rotation" % letter in low:
                        wanted.setdefault(letter, []).append((name, obj))
                # the goniometer angle, where a CODA file keeps it:
                # sample_stick_rotation/value/value (an NXlog inside the
                # NXpositioner).  Mantid does not read logs from there, so it
                # is read here, like the bank angles.  idle_flag and
                # target_value are the motor's other channels, not the angle.
                if ("sample_stick_rotation" in low and "idle_flag" not in low
                        and "target_value" not in low):
                    wanted.setdefault("omega", []).append((name, obj))
            handle.visititems(visit)
            for letter, entries in wanted.items():
                value = None
                for name, obj in entries:
                    if not hasattr(obj, "shape"):
                        continue
                    tail = name.rsplit("/", 1)[-1].lower()
                    if tail not in ("value", "average_value"):
                        continue
                    try:
                        raw = np.atleast_1d(np.asarray(obj[()], dtype=float))
                    except Exception:
                        continue
                    raw = raw[np.isfinite(raw)]
                    if not raw.size:
                        continue
                    if tail == "value" or value is None:
                        value = float(raw[-1] if tail == "value"
                                      else raw.mean())
                    if tail == "value":
                        break
                if value is not None:
                    found[letter] = value
                elif letter != "omega":
                    notes.append("detector_%s_rotation is in the file but"
                                 " carries no number" % letter)
    except Exception as exc:
        return {}, "could not read %s: %s" % (path, exc)
    return found, "; ".join(notes)


def _bank_component(name, letter="a"):
    """The component to rotate, from the file rather than from a guess."""
    if BANK_COMPONENT and letter == "a":
        return BANK_COMPONENT
    instrument = mtd[name].getInstrument()
    for candidate in BANK_COMPONENTS[letter]:
        try:
            if instrument.getComponentByName(candidate) is not None:
                return candidate
        except Exception:
            continue
    names = []
    try:
        for index in range(instrument.nelements()):
            names.append(instrument[index].getName())
    except Exception:
        pass
    say("   no component for bank %s was recognised.  The top" % letter)
    say("   level of this instrument holds: %s"
        % (", ".join(names[:12]) or "nothing this code could read"))
    say("   Pass --bank-component with one of those, or leave --gamma out.")
    return None


def _bank_indices(name):
    """Workspace indices of each bank, from the instrument tree itself."""
    data = mtd[name]
    total = data.getNumberHistograms()
    out, note = {}, ""
    try:
        info = data.componentInfo()
        if data.detectorInfo().size() != total:
            raise ValueError("spectra are not one per detector")
        for letter, candidates in BANK_COMPONENTS.items():
            for candidate in candidates:
                try:
                    index = info.indexOfAny(candidate)
                except Exception:
                    continue
                rows = np.sort(np.array(info.detectorsInSubtree(index),
                                        dtype=int))
                if rows.size:
                    out[letter] = rows
                    break
    except Exception as trouble:
        note = "(%s) " % trouble
        out = {}
    if "a" not in out:
        # sx_magic's layout: bank A is the first 491520 spectra
        out = {"a": np.arange(min(491520, total))}
        if total > 491520:
            out["b"] = np.arange(491520, total)
        note += "banks by spectrum index: A = the first 491520"
    return out, note


def bank_rows(state):
    """The workspace indices of the bank that goes into Q space."""
    banks = state.get("banks") or {}
    if BANK in banks:
        return banks[BANK]
    return np.arange(len(state["two_theta"]))


def bank_theta(state):
    """2theta of the voxels of that bank only."""
    return state["two_theta"][bank_rows(state)]


def frame_gap(name, step=50.0, level=1.0e-3):
    """The empty stretch of the recorded frame, found on the summed events.

    The frame is circular: a file whose events fill 21..71 ms has its gap from
    71 ms through the frame edge to 21 ms, and needs no unwrapping.  A file
    whose band was longer than a frame shifted by the chopper opening has its
    gap INSIDE the frame, between the end of the previous pulse's slow
    neutrons and the start of this pulse's fast ones.
    Returns (centre_us, width_us, crosses_the_frame_edge).
    """
    total = "%s_frame" % PREFIX
    SumSpectra(InputWorkspace=name, OutputWorkspace=total)
    Rebin(InputWorkspace=total, OutputWorkspace=total,
          Params="0,%f,%f" % (step, _FRAME_US), PreserveEvents=False)
    counts = np.array(mtd[total].readY(0), dtype=float)
    DeleteWorkspace(total)
    n = counts.size
    empty = counts <= level*counts.max()
    if empty.all() or not empty.any():
        return None, 0.0, True
    doubled = np.concatenate([empty, empty])
    best, start, run = 0, 0, 0
    for i, flag in enumerate(doubled):
        run = run + 1 if flag else 0
        if run > best and run <= n:
            best, start = run, i - run + 1
    crosses = start + best > n or start == 0 or (start + best) % n == 0
    centre = ((start + 0.5*best) % n)*step
    return centre, best*step, crosses


def unwrap_frames(tag, name):
    """Move the previous pulse's neutrons to t + 71.43 ms (--unwrap)."""
    if not UNWRAP:
        return
    if "t_split" not in SHARED:
        if str(UNWRAP).lower() == "auto":
            centre, width, crosses = frame_gap(name)
            if centre is None or crosses:
                SHARED["t_split"] = 0.0
                say("   %-14s --unwrap auto: the recorded frame has its empty"
                    " stretch across the frame edge, so no event belongs to"
                    " the previous pulse - nothing to unwrap" % tag)
            else:
                SHARED["t_split"] = centre
                say("   %-14s --unwrap auto: empty stretch of %.0f us inside"
                    " the frame, split at %.0f us" % (tag, width, centre))
        else:
            SHARED["t_split"] = 1000.0*float(UNWRAP)
            say("   %-14s --unwrap: split at %.0f us (given)"
                % (tag, SHARED["t_split"]))
    split = SHARED["t_split"]
    if split <= 0:
        return
    before = mtd[name].getNumberEvents()
    slow = "%s_slow" % name
    call("FilterByXValue", InputWorkspace=name, OutputWorkspace=slow,
         XMax=split)
    call("FilterByXValue", InputWorkspace=name, OutputWorkspace=name,
         XMin=split)
    moved = mtd[slow].getNumberEvents()
    ChangeBinOffset(InputWorkspace=slow, OutputWorkspace=slow,
                    Offset=_FRAME_US)
    call("Plus", LHSWorkspace=name, RHSWorkspace=slow, OutputWorkspace=name)
    DeleteWorkspace(slow)
    after = mtd[name].getNumberEvents()
    say("   %-14s unwrapped: %d of %d events (%.1f %%) below %.0f us moved"
        " by +%.1f us - the previous pulse's slow neutrons%s"
        % (tag, moved, before, 100.0*moved/max(before, 1), split, _FRAME_US,
           "" if after == before else
           "   WARNING: %d events before, %d after" % (before, after)))


def _load_one(tag, path, state):
    name = ws(tag, "raw")
    LoadEventNexus(Filename=path, OutputWorkspace=name, LoadMonitors=False)
    unwrap_frames(tag, name)
    data = mtd[name]
    if IDF:
        LoadInstrument(Workspace=name, Filename=IDF, RewriteSpectraMap=False)
    measured, trouble = read_bank_angles(path)
    state["omega"] = measured.pop("omega", None)
    state["measured_angles"] = measured
    state["angle_note"] = trouble
    turns = {}
    for letter, given, carried in (("a", BANK_GAMMA, IDF_GAMMA),
                                   ("b", BANK_GAMMA_B, IDF_GAMMA_B)):
        angle = given if given is not None else measured.get(letter)
        if angle is None or carried is None:
            continue
        # Section 101: Mantid's RotateInstrumentComponent is a right-handed
        # rotation about +y, and a McStas bank angle gamma is the same
        # right-handed rotation (voxelization.py at gamma_d = 0, turned by
        # +10 deg, lies on the 10-deg mesh).  The old '-(gamma - gamma_file)'
        # only worked with MAGiC_Definition_zero.xml, whose own rotation had
        # the opposite sign and cancelled it for runs at the file's 10 deg.
        turn = +(float(angle) - float(carried))
        if abs(turn) < 1e-9:
            turns[letter] = (angle, 0.0, None)
            continue
        component = _bank_component(name, letter)
        if component:
            RotateInstrumentComponent(Workspace=name, ComponentName=component,
                                      X=0, Y=1, Z=0, Angle=turn,
                                      RelativeRotation=True)
        turns[letter] = (angle, turn, component)
    state["turns"] = turns
    helper = ws(tag, "det")
    PreprocessDetectorsToMD(InputWorkspace=name, OutputWorkspace=helper)
    two_theta = np.degrees(np.array(mtd[helper].column("TwoTheta")))
    far = np.array(mtd[helper].column("L2"))
    instrument = data.getInstrument()
    state.update(raw=name, two_theta=two_theta, l2=far,
                 l1=float(instrument.getSource().getDistance(
                     instrument.getSample())))
    run = data.getRun()
    axis = None
    for candidate in ("sample_stick_rotation", "omega", "phi", "chi"):
        if run.hasProperty(candidate):
            axis = candidate
            break
    if axis is None and state.get("omega") is not None:
        # the angle is in the NeXus file but not among Mantid's logs: make it
        # a log, so that SetGoniometer takes it the same way
        call("AddSampleLog", Workspace=name, LogName="omega_nexus",
             LogText="%.6f" % state["omega"], LogType="Number",
             NumberType="Double", LogUnit="deg")
        axis = "omega_nexus"
    state["axis"] = axis
    index_of = {}
    for i in range(data.getNumberHistograms()):
        for detector in data.getSpectrum(i).getDetectorIDs():
            index_of[int(detector)] = i
    state["index_of"] = index_of
    banks, how = _bank_indices(name)
    state["banks"] = banks
    events = np.array([data.getSpectrum(i).getNumberEvents()
                       for i in range(data.getNumberHistograms())])
    say("   %-14s %8d spectra %12d events %7.0f MB"
        % (tag, data.getNumberHistograms(), data.getNumberEvents(), big(name)))
    for letter in sorted(banks):
        rows = banks[letter]
        say("   %-14s bank %s %8d voxels %12d events  2theta %6.2f .. %6.2f%s"
            % ("", letter.upper(), rows.size, int(events[rows].sum()),
               two_theta[rows].min(), two_theta[rows].max(),
               "   <- goes into Q space" if letter == BANK else ""))
    if how:
        say("   %-14s %s" % ("", how))
    if BANK not in banks or not events[banks[BANK]].sum():
        say("   %-14s WARNING: bank %s holds no events in this run"
            % ("", BANK.upper()))
    global T0, T0_SOURCE
    if T0 is None:
        try:
            found = data.getInstrument().getNumberParameter("t0")
        except Exception:
            found = []
        if len(found):
            T0, T0_SOURCE = float(found[0]), "the instrument parameter 't0'"
        else:
            T0, T0_SOURCE = 0.0, "nowhere - no --t0 and no 't0' parameter"
        say("   %-14s t0 = %.0f us, from %s" % ("", T0, T0_SOURCE))
    say("   %-14s tof %8.0f .. %8.0f us   L1 %.4f  L2 %.4f .. %.4f m  %s"
        % ("", data.getTofMin(), data.getTofMax(), state["l1"], far.min(),
           far.max(), "rotation log: %s%s" % (axis, " = %.3f deg, from entry/instrument/sample_stick_rotation" % state["omega"] if axis == "omega_nexus" else "") if axis else "no rotation log"))


def step_load():
    head("1. load %d run%s%s" % (len(RUNS), "" if len(RUNS) == 1 else "s",
                                 " and %s" % NORMALISATION if NORMALISATION
                                 else ""))
    if not RUNS:
        say("   no files given.  magic_workflow.py run1.nxs run2.nxs ...")
        return
    for tag, path, state in each_run():
        _load_one(tag, path, state)
    if IDF:
        say("   geometry in every run replaced by %s" % IDF)
    else:
        say("   geometry taken from each file, as the converter wrote it")
    for tag, _, state in each_run():
        measured = state.get("measured_angles") or {}
        if measured:
            say("   %-14s the file says the banks stood at %s"
                % (tag, ", ".join("%s = %.4f deg" % (k.upper(), v)
                                  for k, v in sorted(measured.items()))))
        elif state.get("angle_note"):
            say("   %-14s %s" % (tag, state["angle_note"]))
        for letter, (angle, turn, component) in sorted(
                (state.get("turns") or {}).items()):
            carried = IDF_GAMMA if letter == "a" else IDF_GAMMA_B
            if abs(turn) < 1e-9:
                say("   %-14s bank %s: at %.4f, geometry carries %.4f, nothing"
                    " turned" % ("", letter.upper(), angle, carried))
            elif component:
                say("   %-14s bank %s: at %.4f, geometry carries %.4f, so %s"
                    " turned %+.4f deg"
                    % ("", letter.upper(), angle, carried, component, turn))
            else:
                say("   %-14s bank %s: NOTHING turned - no component found"
                    % ("", letter.upper()))
        for letter in ("a", "b"):
            if letter not in (state.get("turns") or {}):
                carried = IDF_GAMMA if letter == "a" else IDF_GAMMA_B
                if carried is None:
                    say("   %-14s bank %s left alone: --idf-gamma-%s was not"
                        " given, and a geometry file that does not say what"
                        % ("", letter.upper(), letter))
                    say("   %-14s angle it was built at cannot be corrected"
                        % "")
    shapes = {(state["l1"], len(state["two_theta"])) for _, _, state in each_run()}
    if len(shapes) > 1:
        say("   WARNING: the runs do not share one geometry, so the detector")
        say("   tables and the normalisation cannot be shared either")


# --------------------------------------------------------------- propose --
def _propose_one(tag, state):
    name = state["raw"]
    total = ws(tag, "tof_sum")
    # sum over the spectra FIRST, on the events, and bin afterwards: binning a
    # whole MAGiC bank at 200 us is half a million spectra by a few hundred
    # bins, over a gigabyte, and that is what hangs the interface
    SumSpectra(InputWorkspace=name, OutputWorkspace=total)
    Rebin(InputWorkspace=total, OutputWorkspace=total,
          Params="%f,200,%f" % (mtd[name].getTofMin(), mtd[name].getTofMax()),
          PreserveEvents=False)
    edges = np.array(mtd[total].readX(0), dtype=float)
    counts = np.array(mtd[total].readY(0), dtype=float)
    middle = 0.5*(edges[1:] + edges[:-1])
    inside = counts > 0.02*counts.max() if counts.max() > 0 else counts > 1
    if not inside.any():
        say("   %-14s empty" % tag)
        return None
    first, last = np.flatnonzero(inside)[[0, -1]]
    t_lo, t_hi = float(middle[first]), float(middle[last])
    low = state["l1"] + state["l2"].min()
    high = state["l1"] + state["l2"].max()
    lam = (_CONVERT*(t_lo - T0)/low, _CONVERT*(t_hi - T0)/high)
    flat = ws(tag, "counts")
    call("Integration", InputWorkspace=name, OutputWorkspace=flat)
    per = np.array(mtd[flat].extractY()).ravel()[bank_rows(state)]
    two_theta = bank_theta(state)
    bands = np.linspace(two_theta.min(), two_theta.max(), 61)
    mean = np.array([per[(two_theta >= bands[i]) & (two_theta < bands[i+1])].mean()
                     if ((two_theta >= bands[i]) & (two_theta < bands[i+1])).any()
                     else np.nan for i in range(60)])
    # the plateau is taken over bands that receive anything: bank B holds no
    # events in these files, and its empty bands once outnumbered bank A's and
    # put the plateau at 0, so every band looked like direct beam
    live = mean[np.isfinite(mean) & (mean > 0)]
    plateau = float(np.median(live)) if live.size else 0.0
    # the direct beam is a run of hot bins that STARTS at the lowest angle.
    # Taking the last hot bin anywhere instead once proposed 75.6 deg on a file
    # holding both banks and masked all of bank A.
    cut = None
    if np.isfinite(mean[0]) and mean[0] > 3.0*plateau:
        edge = 0
        while edge + 1 < len(mean) and (not np.isfinite(mean[edge + 1])
                                        or mean[edge + 1] > 3.0*plateau):
            edge += 1
        cut = float(bands[edge + 1])
        share = float(np.mean(two_theta < cut))
        if share > MAX_MASK:
            say("   %-14s a cut at %.1f deg would mask %.0f %% of the voxels,"
                % (tag, cut, 100*share))
            say("   %-14s which is more than %.0f %% - refused, look at the"
                % ("", 100*MAX_MASK))
            say("   %-14s profile below and set --two-theta-min by hand" % "")
            for i in range(0, 14):
                if np.isfinite(mean[i]):
                    say("   %-14s   2theta %6.2f .. %6.2f  %12.0f  %6.1f x"
                        % ("", bands[i], bands[i + 1], mean[i],
                           mean[i]/plateau))
            cut = None
    state.update(t_window=(t_lo, t_hi), lam=lam, cut=cut, plateau=plateau,
                 profile=(bands, mean))
    say("   %-14s frame %8.0f .. %8.0f us (%6.0f us)  lambda %5.3f .. %5.3f A"
        "  %s" % (tag, t_lo, t_hi, t_hi - t_lo, lam[0], lam[1],
                  "direct beam to %4.1f deg" % cut if cut else "no beam edge"))
    if t_hi - t_lo > _FRAME_US:
        say("   %-14s WARNING: longer than one %.0f us frame at 14 Hz, so slow"
            % ("", _FRAME_US))
        say("   %-14s neutrons of one pulse arrive with fast ones of the next"
            % "")
        say("   %-14s and the wavelength of an event is ambiguous" % "")
    return state


def step_propose():
    head("2. propose the cuts, from the data alone")
    if not RUN and not need(step_load, "no run is loaded"):
        return None
    for tag, _, state in each_run():
        _propose_one(tag, state)
    first = RUN.get(tag_of(RUNS[0]), {})
    if first.get("profile"):
        edges, mean = first["profile"]
        say()
        say("   counts per voxel at low angle in %s, against a plateau of %.0f:"
            % (tag_of(RUNS[0]), first["plateau"]))
        for i in range(0, 10):
            if np.isfinite(mean[i]):
                say("     2theta %6.2f .. %6.2f  %12.0f  %6.1f x plateau"
                    % (edges[i], edges[i + 1], mean[i],
                       mean[i]/max(first["plateau"], 1e-9)))
    bands = [state["lam"] for _, _, state in each_run() if state.get("lam")]
    cuts = [state["cut"] for _, _, state in each_run() if state.get("cut")]
    if not bands:
        return None
    common = (max(b[0] for b in bands), min(b[1] for b in bands))
    if LAMBDA_BAND:
        common = LAMBDA_BAND
        say()
        say("   band taken from the command line: %.3f .. %.3f A" % common)
    else:
        say()
        say("   the band every run shares is %.3f .. %.3f A, and that is what"
            % common)
        say("   all of them are cut to, so an intensity from one run means the")
        say("   same as an intensity from another.")
    say("   a fixed time window is a DIFFERENT band in every voxel, because")
    say("   lambda = 3.956e-3 t / L, so the common part is narrower than any")
    say("   single voxel's: the lower bound comes from the shortest flight")
    say("   path at the start of the frame, the upper from the longest at the")
    say("   end.")
    cut = TWO_THETA_MIN if TWO_THETA_MIN is not None else (max(cuts) if cuts
                                                           else None)
    if cut:
        share = float(np.mean(bank_theta(RUN[tag_of(RUNS[0])]) < cut))
        say("   masking below 2theta = %.1f deg in every run: %.1f %% of the"
            % (cut, 100*share))
        say("   voxels.  %s"
            % ("Set --two-theta-min to override." if share < MAX_MASK
               else "That is a lot - check it."))
    else:
        say("   no angular mask.  If the peak search finds nothing but noise")
        say("   near |Q| = 0, set --two-theta-min by hand from the profile")
        say("   above.")
    SHARED["lam"] = common
    SHARED["cut"] = cut
    say()
    confirm("lambda %.3f..%.3f A, 2theta > %s"
            % (common[0], common[1], "%.1f deg" % cut if cut else "no cut"))
    return common


# --------------------------------------------------------------- prepare --
def step_prepare():
    head("3. apply the cuts and the goniometer")
    if "lam" not in SHARED and not need(step_propose, "no band was proposed"):
        return None
    lam, cut = SHARED["lam"], SHARED["cut"]
    for tag, _, state in each_run():
        _prepare_one(tag, state, lam, cut)
    if not any(state.get("axis") for _, _, state in each_run()):
        say("   no run carries a rotation log, so each orientation found below")
        say("   is that of one setting and the runs are independent settings")


def _prepare_one(tag, state, lam, cut, quiet=False):
    if state.get("data") and state["data"] in mtd:
        return state["data"]
    if True:
        name = ws(tag, "data")
        CropWorkspace(InputWorkspace=state["raw"], OutputWorkspace=name,
                      XMin=state["t_window"][0], XMax=state["t_window"][1])
        before = mtd[name].getNumberEvents()
        if T0:
            ChangeBinOffset(InputWorkspace=name, OutputWorkspace=name,
                            Offset=-float(T0))
        ConvertUnits(InputWorkspace=name, OutputWorkspace=name,
                     Target="Wavelength", EMode="Elastic")
        CropWorkspace(InputWorkspace=name, OutputWorkspace=name,
                      XMin=lam[0], XMax=lam[1])
        ConvertUnits(InputWorkspace=name, OutputWorkspace=name, Target="TOF",
                     EMode="Elastic")
        after = mtd[name].getNumberEvents()
        if state.get("axis"):
            SetGoniometer(Workspace=name, Axis0="%s,0,1,0,1" % state["axis"])
        else:
            SetGoniometer(Workspace=name, Axis0="0,0,1,0,1")
        masked = 0
        rows = bank_rows(state)
        other = np.setdiff1d(np.arange(len(state["two_theta"])), rows)
        low = rows[state["two_theta"][rows] < cut] if cut else rows[:0]
        drop = np.concatenate([other, low])
        if drop.size:
            MaskDetectors(Workspace=name,
                          WorkspaceIndexList=[int(i) for i in drop])
        masked = low.size
        if other.size and not quiet:
            say("   %-14s the other bank, %d voxels, masked: only bank %s goes"
                " into Q space" % (tag, other.size, BANK.upper()))
        state["data"] = name
        if not quiet:
            say("   %-14s %12d -> %12d events (%.1f %%), %6d voxels masked, %s"
                % (tag, before, after, 100.0*after/max(before, 1), masked,
                   "goniometer from %s" % state["axis"] if state.get("axis")
                   else "goniometer at identity"))
        return name


# ----------------------------------------------------------------- peaks --
def _build_md(tag, state, lam):
    """Convert one run to Q space.  The caller drops it again when done.

    Only one of these lives at a time.  Three runs of a full MAGiC bank, each
    with its raw workspace, its prepared workspace and its MD boxes, do not fit
    in a laptop's memory, and what that looks like from outside is a freeze in
    whatever algorithm happens to be running.
    """
    sin_max = np.sin(np.radians(bank_theta(state).max()/2.0))
    reach = 4*np.pi*sin_max/lam[0]
    box = 1.03*reach
    md = ws(tag, "md")
    tick("ConvertToMD %s, box +-%.3f A-1 (the bank reaches %.3f at %.2f A)"
         % (tag, box, reach, lam[0]))
    call("ConvertToMD", InputWorkspace=state["data"], OutputWorkspace=md,
         QDimensions="Q3D", dEAnalysisMode="Elastic", Q3DFrames="Q_sample",
         MinValues=[-box, -box, -box], MaxValues=[box, box, box],
         SplitInto=5, SplitThreshold=200, MaxRecursionDepth=7,
         OverwriteExisting=True)
    state["md"] = md
    state["box"] = box
    tick("%s is %.0f MB, finest cell %.5f A-1" % (md, big(md), 2*box/5**7))
    return md


def _release(tag, state, keep_peaks=True):
    """Give the memory back, so the next run has room."""
    for key in ("md", "data"):
        name = state.get(key)
        if name and name in mtd and not (key == "md" and KEEP_MD):
            size = big(name)
            DeleteWorkspace(name)
            state[key] = None
            tick("dropped %s (%.0f MB)" % (name, size))


def step_peaks():
    head("4. find the peaks in every run, one run at a time")
    if not any(state.get("data") for _, _, state in each_run()):
        if not need(step_prepare, "no run has been prepared"):
            return None
    lam = SHARED["lam"]
    say("   one run is held in Q space at a time and let go before the next,")
    say("   because three MD workspaces of a full bank will not fit beside")
    say("   three raw ones.  The peak list and the peak widths are all that is")
    say("   kept; the MD is rebuilt for the integration.")
    for tag, _, state in each_run():
        say()
        say("   %s" % tag)
        _prepare_one(tag, state, lam, SHARED["cut"], quiet=True)
        _build_md(tag, state, lam)
        found = {}
        tick("scanning the density threshold")
        for factor in DENSITY_LADDER:
            scan = ws(tag, "scan")
            call("FindPeaksMD", InputWorkspace=state["md"],
                 OutputWorkspace=scan, DensityThresholdFactor=float(factor),
                 PeakDistanceThreshold=PEAK_DISTANCE,
                 MaxPeaks=2000, OutputType="LeanElasticPeak")
            found[factor] = mtd[scan].getNumberPeaks()
            DeleteWorkspace(scan)
        say("   %-14s ladder: %s"
            % ("", "  ".join("%d:%d" % (f, found[f]) for f in DENSITY_LADDER)))
        usable = [f for f in DENSITY_LADDER if MIN_PEAKS <= found[f] <= MAX_PEAKS]
        choice = usable[0] if usable else min(
            DENSITY_LADDER, key=lambda f: abs(found[f] - MAX_PEAKS))
        if found[DENSITY_LADDER[0]] >= 2000:
            say("   %-14s the loose end of the ladder is capped at 2000, so"
                % "")
            say("   %-14s most of what it finds there is noise" % "")
        peaks = ws(tag, "peaks")
        tick("FindPeaksMD at factor %d" % choice)
        call("FindPeaksMD", InputWorkspace=state["md"], OutputWorkspace=peaks,
             DensityThresholdFactor=float(choice),
             PeakDistanceThreshold=PEAK_DISTANCE, MaxPeaks=2000,
             OutputType="LeanElasticPeak")
        # LeanElasticPeak, as sx_magic: the peak is its Q and nothing else.  A
        # full Peak ray-traces for a detector on a 0.5 m deep bank of 491520
        # voxels, which is slow and puts nothing into indexing that Q does
        # not already carry.  The reflections that are integrated are
        # PREDICTED later, and those do get their detectors.
        tick("CentroidPeaksMD on %d peaks" % mtd[peaks].getNumberPeaks())
        call("CentroidPeaksMD", InputWorkspace=state["md"],
             PeaksWorkspace=peaks, OutputWorkspace=peaks, PeakRadius=0.15)
        table = mtd[peaks]
        q = np.array([list(p.getQSampleFrame()) for p in table])
        counts = np.array([p.getBinCount() for p in table])
        if not counts.any():
            counts = np.array([p.getIntensity() for p in table])
        state.update(peaks=peaks, q=q, counts=counts, density=choice)
        if len(q) == 0:
            say("   %-14s NO PEAKS at any threshold.  Look first at how much" % "")
            say("   %-14s the angular mask took in step 2 and at the band in" % "")
            say("   %-14s step 3: an empty list almost always means the data" % "")
            say("   %-14s were cut away, not that there are no reflections." % "")
            _release(tag, state)
            continue
        norm = np.linalg.norm(q, axis=1)
        say("   %-14s factor %5d -> %4d peaks, |Q| %.3f .. %.3f, d %.3f .. %.3f A"
            % ("", choice, len(q), norm.min(), norm.max(), 2*np.pi/norm.max(),
               2*np.pi/norm.min()))
        tick("measuring the shape of the %d strongest peak clouds"
             % min(RESOLUTION_PEAKS, len(q)))
        rows = _widths(state["md"], peaks, RESOLUTION_PEAKS)
        if len(rows):
            state["widths"] = rows
            say("   %-14s %d widths measured; sigma along Q %.4f .. %.4f,"
                % ("", len(rows), rows[:, 1].min(), rows[:, 1].max()))
            say("   %-14s across %.4f .. %.4f"
                % ("", min(rows[:, 2].min(), rows[:, 3].min()),
                   max(rows[:, 2].max(), rows[:, 3].max())))
        else:
            say("   %-14s no reflection was strong enough to measure" % "")
        _release(tag, state)
        confirm("keep %d peaks from %s" % (len(q), tag))


# --------------------------------------------------- shape, from sx_magic.py --
# The cube-plus-3D-covariance estimator that was here measured the box, not the
# peak.  A +-0.25 A^-1 cube binned 25^3 has bins of 0.020 A^-1 while the true
# transverse sigma is 0.014 to 0.033, so the peak occupied less than one bin;
# and a uniformly filled cube returns sigma = 0.5/sqrt(12) = 0.1443 by itself,
# which is why the fit came back with 0.08 to 0.15 and NEGATIVE slopes.  What
# follows is lifted from sx_magic.py, where it is validated on these files.
SHAPE_HALF = (0.35, 0.09, 0.09)     # half-widths of the box, in the local frame
SHAPE_BINS = (71, 37, 37)           # 0.0099 A^-1 along Q, 0.0049 across
SHAPE_CORE = (0.08, 0.025)          # the core of the other two axes to sum over
SHAPE_MIN_RATIO = 8.0               # I/sigma below this is not worth fitting
SHAPE_MAX_PEAKS = 25
SHAPE_PROBE = (0.07, 0.09, 0.14)    # the spherical probe that ranks the peaks


def local_frame(q):
    """Right-handed frame at Q: along Q, then two directions across it."""
    radial = np.asarray(q, dtype=float)
    radial = radial/np.linalg.norm(radial)
    up = np.array([0.0, 1.0, 0.0])
    across = np.cross(up, radial)
    if np.linalg.norm(across) < 1.0e-6:
        across = np.array([1.0, 0.0, 0.0])
    across = across/np.linalg.norm(across)
    return radial, across, np.cross(radial, across)


def profile_width(axis, profile, window=2.5):
    """Width of a profile after a flat background, by a walking window.

    A plain second moment of the whole box is inflated by whatever pedestal the
    flat background leaves, because a moment weights by (x - centre)^2 and a few
    stray counts far out count for a great deal.  So the centre and width are
    re-estimated inside +-window sigma until they stop moving, and the result is
    corrected for the truncation the window imposes.  A half-maximum width was
    tried first and had to be dropped: at these bin sizes it walks out from the
    single highest bin and a noise spike stops it at one or two bins.
    """
    profile = np.asarray(profile, dtype=float)
    axis = np.asarray(axis, dtype=float)
    edge = max(2, len(profile)//6)
    base = float(np.median(np.concatenate([profile[:edge], profile[-edge:]])))
    clean = profile - base
    clean[~np.isfinite(clean)] = 0.0
    clean[clean < 0.0] = 0.0
    total = clean.sum()
    if total <= 0.0:
        return np.nan
    centre = float((axis*clean).sum()/total)
    variance = float((clean*(axis - centre)**2).sum()/total)
    if not np.isfinite(variance) or variance <= 0.0:
        return np.nan
    k = float(window)
    phi = np.exp(-0.5*k*k)/np.sqrt(2.0*np.pi)
    shrink = 1.0 - 2.0*k*phi/math.erf(k/np.sqrt(2.0))
    step = float(axis[1] - axis[0])
    middle, width = centre, float(np.sqrt(variance))
    for _ in range(8):
        inside = np.abs(axis - middle) <= k*width
        if inside.sum() < 5:
            return np.nan
        outside = np.abs(axis - middle) > k*width + 3.0*step
        level = float(np.median(profile[outside])) if outside.sum() >= 4 else base
        local = profile[inside] - level
        local[local < 0.0] = 0.0
        weight = local.sum()
        if weight <= 0.0:
            return np.nan
        new_middle = float((axis[inside]*local).sum()/weight)
        new_variance = float((local*(axis[inside] - new_middle)**2).sum()/weight)
        if new_variance <= 0.0:
            return np.nan
        new_width = float(np.sqrt(new_variance/shrink))
        settled = (abs(new_width - width) < 0.02*width
                   and abs(new_middle - middle) < 0.2*step)
        middle, width = new_middle, new_width
        if settled:
            break
    return width


def _widths(md, peaks, wanted):
    """Measure sigma along Q and across it, on the strongest reflections.

    The box is attached to each reflection - along Q and twice across - so the
    two directions are never mixed, and each of the three profiles is summed
    over the CORE of the other two axes only.  That last detail is the one that
    matters: the far corners of a box are background, and they would swamp a
    peak 0.025 A^-1 wide.
    """
    table = mtd[peaks]
    if table.getNumberPeaks() == 0:
        return np.zeros((0, 4))
    probe = "%s_probe" % PREFIX
    call("CloneWorkspace", InputWorkspace=peaks, OutputWorkspace=probe)
    call("IntegratePeaksMD", InputWorkspace=md, PeaksWorkspace=probe,
         OutputWorkspace=probe, PeakRadius=SHAPE_PROBE[0],
         BackgroundInnerRadius=SHAPE_PROBE[1],
         BackgroundOuterRadius=SHAPE_PROBE[2], IntegrateIfOnEdge=True)
    strong = []
    for index in range(mtd[probe].getNumberPeaks()):
        peak = mtd[probe].getPeak(index)
        error = peak.getSigmaIntensity()
        if error > 0.0 and peak.getIntensity()/error > SHAPE_MIN_RATIO:
            strong.append((peak.getIntensity()/error, index))
    strong.sort(reverse=True)
    picked = [index for _, index in strong[:min(wanted, SHAPE_MAX_PEAKS)]]
    say("   %-14s %d of %d reflections above I/sigma %.0f; the shape is"
        % ("", len(picked), mtd[probe].getNumberPeaks(), SHAPE_MIN_RATIO))
    say("   %-14s measured on %d of them" % ("", len(picked)))
    DeleteWorkspace(probe)
    if not picked:
        return np.zeros((0, 4))
    axes = [np.linspace(-h, h, n + 1)[:-1] + h/n
            for h, n in zip(SHAPE_HALF, SHAPE_BINS)]
    radial_core = np.abs(axes[0]) < SHAPE_CORE[0]
    across_core = [np.abs(a) < SHAPE_CORE[1] for a in axes[1:]]
    box = "%s_box" % PREFIX
    rows = []
    for index in picked:
        peak = table.getPeak(index)
        q = np.array(list(peak.getQSampleFrame()))
        modulus = float(np.linalg.norm(q))
        if modulus <= 0.0:
            continue
        radial, first, second = local_frame(q)
        try:
            call("BinMD", InputWorkspace=md, OutputWorkspace=box,
                 AxisAligned=False,
                 BasisVector0="r,A^-1,%.8f,%.8f,%.8f" % tuple(radial),
                 BasisVector1="t1,A^-1,%.8f,%.8f,%.8f" % tuple(first),
                 BasisVector2="t2,A^-1,%.8f,%.8f,%.8f" % tuple(second),
                 NormalizeBasisVectors=False,
                 Translation="%.8f,%.8f,%.8f" % tuple(q),
                 OutputExtents=[-SHAPE_HALF[0], SHAPE_HALF[0],
                                -SHAPE_HALF[1], SHAPE_HALF[1],
                                -SHAPE_HALF[2], SHAPE_HALF[2]],
                 OutputBins=list(SHAPE_BINS))
            signal = np.array(mtd[box].getSignalArray(), dtype=float)
        finally:
            if box in mtd:
                DeleteWorkspace(box)
        along = np.nansum(signal[:, across_core[0], :][:, :, across_core[1]],
                          axis=(1, 2))
        one = np.nansum(signal[radial_core, :, :][:, :, across_core[1]],
                        axis=(0, 2))
        two = np.nansum(signal[radial_core, :, :][:, across_core[0], :],
                        axis=(0, 1))
        sigma_r = profile_width(axes[0], along)
        sigma_1 = profile_width(axes[1], one)
        sigma_2 = profile_width(axes[2], two)
        if np.isfinite([sigma_r, sigma_1, sigma_2]).all():
            rows.append((modulus, sigma_r, sigma_1, sigma_2))
    return np.array(rows) if rows else np.zeros((0, 4))


def step_resolution():
    head("5. the resolution function - one instrument, so one function")
    if not any(state.get("widths") is not None for _, _, state in each_run()):
        if not need(step_peaks, "no run has measured peak widths"):
            return None
    gathered = []
    for tag, _, state in each_run():
        if state.get("widths") is None or not len(state["widths"]):
            say("   %-14s no usable peak clouds" % tag)
            continue
        gathered.append(state["widths"])
        fit_r = fit_linear(state["widths"][:, 0], state["widths"][:, 1])
        across = np.maximum(state["widths"][:, 2], state["widths"][:, 3])
        fit_t = fit_linear(state["widths"][:, 0], across)
        say("   %-14s %3d clouds; along Q %s; across %s"
            % (tag, len(state["widths"]), sigma_words(fit_r),
               sigma_words(fit_t)))
    if not gathered:
        say("   nothing to fit")
        return None
    data = np.vstack(gathered)
    # Section 113: ONE resolution function for indexing and integration, as
    # sx_magic measured it - sigma = a + b|Q| along Q, and across Q the WIDER
    # of the two transverse directions (a MAGiC peak is not round across Q,
    # and a width must hold the peak, which the mean of two does not).
    across = np.maximum(data[:, 2], data[:, 3])
    fit_r = fit_linear(data[:, 0], data[:, 1])
    fit_t = fit_linear(data[:, 0], across)
    if fit_r[1] > RADIAL_CONSTANT_MAX:
        # section 86: a constant term badly determined by clouds at |Q| 4-9
        # once blew the indexing windows open at the seeds
        say("   the radial constant came out %.4f A-1 (> %.3f): not trusted,"
            " the radial width is taken proportional to |Q|"
            % (fit_r[1], RADIAL_CONSTANT_MAX))
        fit_r = ("lin", 0.0, float(np.dot(data[:, 0], data[:, 1])
                                   / np.dot(data[:, 0], data[:, 0])))
    spread_r = float(np.std(data[:, 1] - sigma_of(fit_r, data[:, 0])))
    spread_t = float(np.std(across - sigma_of(fit_t, data[:, 0])))
    middle = float(data[:, 0].mean())
    say()
    say("   all %d clouds together - the resolution belongs to the instrument,"
        % len(data))
    say("   not to a run, so this is the function used from here on:")
    say("     sigma along Q  = %s   scatter %.4f" % (sigma_words(fit_r),
                                                       spread_r))
    say("     sigma across Q = %s   scatter %.4f   (the wider transverse"
        " direction)" % (sigma_words(fit_t), spread_t))
    say("   at |Q| = %.2f the ratio is %.1f : 1.  That is why a spherical"
        % (middle, float(sigma_of(fit_r, middle))
           / max(float(sigma_of(fit_t, middle)), 1e-9)))
    say("   integration radius must either miss signal or take in background,")
    say("   and why an isotropic indexing tolerance fails on this instrument.")
    SHARED["sigma_radial"] = fit_r
    SHARED["sigma_across"] = fit_t
    if spread_r > 0.5*float(sigma_of(fit_r, middle)) or \
            spread_t > 0.5*float(sigma_of(fit_t, middle)):
        say()
        say("   WARNING: the scatter is more than half the width itself, so")
        say("   this is not a measurement of the instrument.  The widths that")
        say("   follow, the indexing tolerance and every integration volume")
        say("   are all built on it, so look at step 4 and at the box the")
        say("   widths were measured in before believing anything below.")
    return fit_r, fit_t


def sigma_of(pair, norm):
    """sigma = sqrt(a^2 + (b |Q|)^2).

    Independent contributions to a width add in quadrature, not linearly, and
    the quadrature form is positive whatever the fit does.  A straight line in
    sigma was tried first and gave a NEGATIVE transverse width above |Q| = 9.7,
    which then went into an integration radius.
    """
    if len(pair) == 3:                       # ("lin", a, b): a + b |Q|
        _, a, b = pair
        return a + b*np.asarray(norm, dtype=float)
    a, b = pair
    return np.sqrt(a*a + (b*np.asarray(norm, dtype=float))**2)


def fit_linear(norm, width):
    """sigma = a + b|Q| with a, b >= 0 (section 113; sx_magic's form).

    Non-negative, so the negative widths that the first straight-line fit gave
    above |Q| = 9.7 cannot happen: a zero coefficient is refitted alone.
    """
    norm = np.asarray(norm, dtype=float)
    width = np.asarray(width, dtype=float)
    design = np.vstack([np.ones_like(norm), norm]).T
    (a, b), *_ = np.linalg.lstsq(design, width, rcond=None)
    if a < 0:
        a, b = 0.0, float(np.dot(norm, width)/max(np.dot(norm, norm), 1e-30))
    if b < 0:
        a, b = float(np.mean(width)), 0.0
    return ("lin", float(a), float(b))


def sigma_words(pair):
    if len(pair) == 3:
        return "%.4f + %.4f |Q|" % (pair[1], pair[2])
    return "sqrt(%.4f^2 + (%.4f |Q|)^2)" % pair


def fit_sigma(norm, width, proportional=False):
    """Least squares for sigma^2 = a^2 + (b |Q|)^2, both kept positive.

    proportional=True fits b alone (a = 0), which is what the RADIAL width
    gets.  On a TOF instrument the width along Q is dominated by dlambda/lambda
    and dtheta cot(theta), i.e. a fixed fraction of |Q|.  The clouds that can be
    measured sit at |Q| = 4 to 9, so a free constant term is barely determined
    there and extrapolates wildly to |Q| < 3.2, where the indexing takes its
    seeds: on 28 Sep one run gave a = 0.088, which doubled the radial window
    at the seeds and made runs 2 and 3 unindexable on peak lists that index
    at 66 and 35 with a = 0.  Checked offline on sx_magic's own peak lists.
    """
    if proportional:
        norm = np.asarray(norm, dtype=float)
        target = np.asarray(width, dtype=float)**2
        b2 = float(np.sum(norm**2*target)/max(float(np.sum(norm**4)), 1e-30))
        return 1e-4, float(np.sqrt(max(b2, 1e-12)))
    # Non-negative least squares for two parameters, done properly: when the
    # free fit wants a negative term, that term is set to zero and the OTHER
    # one is refitted alone.  Clamping without refitting (the first version)
    # kept the slope of a fit that had used a negative intercept to get it, so
    # the width came out several times too small at low |Q| - exactly where
    # the indexing seeds are taken - and nothing indexed.
    norm = np.asarray(norm, dtype=float)
    target = np.asarray(width, dtype=float)**2
    design = np.vstack([np.ones_like(norm), norm**2]).T
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    if solution[0] < 0 or solution[1] < 0:
        only_b = float(np.dot(norm**2, target)/max(float(np.sum(norm**4)), 1e-30))
        only_a = float(np.mean(target))
        cost_b = float(np.sum((target - only_b*norm**2)**2))
        cost_a = float(np.sum((target - only_a)**2))
        solution = (0.0, max(only_b, 0.0)) if cost_b < cost_a else \
                   (max(only_a, 0.0), 0.0)
    return (float(np.sqrt(max(solution[0], 1e-8))),
            float(np.sqrt(max(solution[1], 1e-12))))


def sigmas(norm):
    return (sigma_of(SHARED["sigma_radial"], norm),
            sigma_of(SHARED["sigma_across"], norm))


# -------------------------------------------------------------- the lattice --
FREE = {"triclinic": 6, "monoclinic": 4, "orthorhombic": 3, "tetragonal": 2,
        "rhombohedral": 2, "hexagonal": 2, "cubic": 1}


def cell_to_basis(cell):
    """Reciprocal basis, carrying the 2 pi, from a b c alpha beta gamma."""
    a, b, c, al, be, ga = cell
    al, be, ga = np.radians([al, be, ga])
    volume = a*b*c*np.sqrt(max(1 - np.cos(al)**2 - np.cos(be)**2
                               - np.cos(ga)**2
                               + 2*np.cos(al)*np.cos(be)*np.cos(ga), 1e-12))
    direct = np.array([
        [a, b*np.cos(ga), c*np.cos(be)],
        [0.0, b*np.sin(ga), c*(np.cos(al) - np.cos(be)*np.cos(ga))/np.sin(ga)],
        [0.0, 0.0, volume/(a*b*np.sin(ga))]])
    return 2*np.pi*np.linalg.inv(direct).T


def basis_to_cell(basis):
    direct = np.linalg.inv(basis/(2*np.pi)).T
    vectors = [direct[:, i] for i in range(3)]
    lengths = [float(np.linalg.norm(v)) for v in vectors]
    angles = []
    for i, j in ((1, 2), (0, 2), (0, 1)):
        cosine = (vectors[i] @ vectors[j])/(lengths[i]*lengths[j])
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
    return lengths + angles


def pack(cell, system):
    a, b, c, al, be, ga = cell
    return {"triclinic": [a, b, c, al, be, ga], "monoclinic": [a, b, c, be],
            "orthorhombic": [a, b, c], "tetragonal": [0.5*(a + b), c],
            "rhombohedral": [(a + b + c)/3.0, (al + be + ga)/3.0],
            "hexagonal": [0.5*(a + b), c], "cubic": [(a + b + c)/3.0]}[system]


def unpack(values, system):
    v = list(values)
    if system == "triclinic":
        return v
    if system == "monoclinic":
        return [v[0], v[1], v[2], 90.0, v[3], 90.0]
    if system == "orthorhombic":
        return [v[0], v[1], v[2], 90.0, 90.0, 90.0]
    if system == "tetragonal":
        return [v[0], v[0], v[1], 90.0, 90.0, 90.0]
    if system == "rhombohedral":
        return [v[0], v[0], v[0], v[1], v[1], v[1]]
    if system == "hexagonal":
        return [v[0], v[0], v[1], 90.0, 90.0, 120.0]
    return [v[0], v[0], v[0], 90.0, 90.0, 90.0]


def rotation(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = np.asarray(vector)/angle
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)


def spin_of(matrix):
    """The rotation vector closest to taking the identity to this matrix."""
    u, _, vt = np.linalg.svd(matrix)
    spin = u @ vt
    angle = float(np.arccos(np.clip((np.trace(spin) - 1)/2.0, -1, 1)))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([spin[2, 1] - spin[1, 2], spin[0, 2] - spin[2, 0],
                     spin[1, 0] - spin[0, 1]])
    size = float(np.linalg.norm(axis))
    if size < 1e-6:
        # at 180 deg the antisymmetric part vanishes; the axis is then the
        # eigenvector of R with eigenvalue +1 (the first version returned a
        # zero vector here, i.e. called a half turn "no rotation")
        values, vectors = np.linalg.eig(spin)
        axis = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
        size = float(np.linalg.norm(axis))
    return angle*axis/size


def weighted_residual(q, hkl, matrix):
    delta = q - (matrix @ np.asarray(hkl, dtype=float).T).T
    norm = np.linalg.norm(q, axis=1)
    unit = q/np.maximum(norm, 1e-12)[:, None]
    radial = (delta*unit).sum(axis=1)
    across = np.linalg.norm(delta - radial[:, None]*unit, axis=1)
    sig_r, sig_t = sigmas(norm)
    return radial/np.maximum(sig_r, 1e-9), across/np.maximum(sig_t, 1e-9)


def reduce_basis(matrix, rounds=40):
    columns = [matrix[:, i].copy() for i in range(3)]
    for _ in range(rounds):
        columns.sort(key=lambda v: np.linalg.norm(v))
        changed = False
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                step = int(round((columns[i] @ columns[j])
                                 / max(columns[j] @ columns[j], 1e-12)))
                if step and np.linalg.norm(columns[i] - step*columns[j]) \
                        < np.linalg.norm(columns[i]) - 1e-9:
                    columns[i] = columns[i] - step*columns[j]
                    changed = True
        if not changed:
            break
    return np.array(columns).T


def _reflections(cell, q_min, q_max, limit=6000):
    """Every hkl of this cell with |G| inside the range the data reach."""
    basis = cell_to_basis(cell)
    reach = [int(np.ceil(q_max/np.linalg.norm(basis[:, i]))) + 1
             for i in range(3)]
    h = np.arange(-reach[0], reach[0] + 1)
    k = np.arange(-reach[1], reach[1] + 1)
    l = np.arange(-reach[2], reach[2] + 1)
    grid = np.array(np.meshgrid(h, k, l, indexing="ij")).reshape(3, -1).T
    grid = grid[np.abs(grid).sum(axis=1) > 0]
    vectors = (basis @ grid.T).T
    size = np.linalg.norm(vectors, axis=1)
    keep = (size >= q_min) & (size <= q_max)
    grid, vectors, size = grid[keep], vectors[keep], size[keep]
    if len(grid) > limit:
        order = np.argsort(size)[:limit]
        grid, vectors, size = grid[order], vectors[order], size[order]
    return grid, vectors, size


# ------------------------------------------------ indexing, from sx_magic.py --
# np.rint was the wrong nearest-node rule.  It picks the nearest node under the
# CELL metric, but the radial coordinate of a peak is the inaccurate one: with
# sigma_r about 0.04 + 0.0115|Q| the 3 sigma radial window at |Q| = 8 is +-0.4
# A^-1 against c* = 0.4995, so a peak whose true node is one step of l away is
# assigned whichever node rounding lands on and then passes the residual test.
# So: choose the node by DIRECTION, among all 27 neighbours, allowing |Q| to be
# wrong by INDEX_SIGMA_R radial sigmas, and judge radial and transverse
# separately - which is the whole point.
INDEX_SIGMA_R = 3.0             # radial window, in sigmas of the resolution
INDEX_SIGMA_T = 2.0             # transverse window
INDEX_CAP = 9.0                 # a 3-sigma cap on one peak's contribution
INDEX_ANGLE_TOL = 0.010         # cosine tolerance on a pair angle, about 0.6 deg
INDEX_SEED_QMAX = 3.2           # seeds come from low |Q|, where a length means
                                # something: at high |Q| hundreds of
                                # reflections match one length within 3 sigma
INDEX_ROUNDS = 3                # index -> refine the rotation -> re-index
_NEIGHBOURS = np.array([(h, k, l) for h in (-1, 0, 1) for k in (-1, 0, 1)
                        for l in (-1, 0, 1)], dtype=float)


def assign_indices(q, matrix):
    """Nearest lattice node by direction, letting |Q| be wrong by 3 sigma_r."""
    if abs(np.linalg.det(matrix)) < 1.0e-12:
        raise ValueError("the reciprocal basis collapsed")
    norm = np.linalg.norm(q, axis=1)
    direction = q/np.maximum(norm, 1e-12)[:, None]
    sig_r, _ = sigmas(norm)
    base = np.round(np.linalg.solve(matrix, q.T).T)
    best = np.full(len(q), np.inf)
    chosen = base.copy()
    for step in _NEIGHBOURS:
        trial = base + step
        offset = (matrix @ trial.T).T - q
        radial = (offset*direction).sum(axis=1)
        across = np.linalg.norm(offset - radial[:, None]*direction, axis=1)
        penalty = np.where(np.abs(radial) < INDEX_SIGMA_R*sig_r, across, np.inf)
        take = penalty < best
        best[take], chosen[take] = penalty[take], trial[take]
    offset = (matrix @ chosen.T).T - q
    radial = (offset*direction).sum(axis=1)
    across = np.linalg.norm(offset - radial[:, None]*direction, axis=1)
    return chosen.astype(int), radial, across


def count_indexed(q, matrix, tolerance=None):
    try:
        hkl, radial, across = assign_indices(q, matrix)
    except (ValueError, np.linalg.LinAlgError):
        return 0, None, None
    sig_r, sig_t = sigmas(np.linalg.norm(q, axis=1))
    good = ((np.abs(radial) < INDEX_SIGMA_R*sig_r)
            & (across < INDEX_SIGMA_T*sig_t)
            & (np.abs(hkl).sum(axis=1) > 0))
    return int(good.sum()), hkl, good


def robust_score(q, matrix):
    """Mean capped Mahalanobis distance - lower is better.

    Counting the peaks that fall inside a tolerance is monotone in the
    tolerance and in the cell volume, so it rewards a looser fit and a denser
    lattice.  A capped sum of (dr/sigma_r)^2 + (dt/sigma_t)^2 does not: a peak
    that is nowhere near a node contributes the cap whatever happens, and a
    denser lattice buys nothing.
    """
    try:
        _, radial, across = assign_indices(q, matrix)
    except (ValueError, np.linalg.LinAlgError):
        return np.inf
    sig_r, sig_t = sigmas(np.linalg.norm(q, axis=1))
    return float(np.minimum((radial/sig_r)**2 + (across/sig_t)**2,
                            INDEX_CAP).mean())


def _chance(q, matrix):
    """How many matches a random peak list would have produced.

    The acceptance region is a cylinder of radius INDEX_SIGMA_T sigma_t and
    length 2 INDEX_SIGMA_R sigma_r, and the density of nodes is 1/|det|.
    """
    sig_r, sig_t = sigmas(np.linalg.norm(q, axis=1))
    volume = np.pi*(INDEX_SIGMA_T*sig_t)**2*(2.0*INDEX_SIGMA_R*sig_r)
    return float((volume/abs(np.linalg.det(matrix))).sum())


def refine_rotation(q, hkl, cell, good):
    """The rotation that best carries the lattice onto the peaks, cell fixed.

    Weighted so that the transverse offset, which is the accurate coordinate,
    decides: each matched peak is weighted by 1/sigma_t^2.
    """
    if good.sum() < 4:
        return None
    basis = cell_to_basis(cell)
    target = q[good]
    source = (basis @ hkl[good].astype(float).T).T
    _, sig_t = sigmas(np.linalg.norm(target, axis=1))
    weight = 1.0/np.maximum(sig_t, 1e-9)**2
    cross = (source*weight[:, None]).T @ target
    u, _, vt = np.linalg.svd(cross)
    spin = (u @ vt).T
    if np.linalg.det(spin) < 0:
        u[:, -1] = -u[:, -1]
        spin = (u @ vt).T
    return spin @ basis


def _orientation(tag, state, cell, budget=40000):
    """Find the rotation that puts a KNOWN lattice onto the observed peaks.

    Two peaks and two reflections fix a rotation, so the search is over pairs:
    match both lengths and the angle between them, build the rotation, and score
    it on the whole lattice.  Seeds are taken at low |Q| on purpose.
    """
    q, counts = state["q"], state["counts"]
    if len(q) < 8:
        say("   %-14s only %d peaks" % (tag, len(q)))
        return None
    norm = np.linalg.norm(q, axis=1)
    grid, vectors, size = _reflections(cell, norm.min()*0.97, norm.max()*1.03)
    if CENTRING and CENTRING.upper() != "P":
        keep = allowed_by_centring(grid, CENTRING)
        grid, vectors, size = grid[keep], vectors[keep], size[keep]
    if not len(grid):
        say("   %-14s the given cell has no reflection in |Q| %.2f .. %.2f"
            % (tag, norm.min(), norm.max()))
        return None
    sig_r, sig_t = sigmas(norm)
    # As sx_magic's stage `index`, which is what works on these data:
    # * the seeds are the 11 brightest peaks below INDEX_SEED_QMAX, the window
    #   widened until they carry enough candidate reflections between them;
    # * a trial rotation is judged on the 80 BRIGHTEST peaks only, by how many
    #   of them it indexes.  Scoring on every peak - 500 to 800 of which most
    #   are background clumps - drowned the real ones: a mean over all peaks
    #   is decided by the noise, and runs 2 and 3 never indexed.  With the cell
    #   given, the lattice density is fixed, so a count cannot be bought with a
    #   denser lattice here.
    order = np.array([], dtype=int)
    for limit in (INDEX_SEED_QMAX, 1.5*INDEX_SEED_QMAX, 2.2*INDEX_SEED_QMAX,
                  np.inf):
        pool = np.flatnonzero(norm < limit)
        if pool.size < 2:
            continue
        order = pool[np.argsort(-counts[pool])][:11]
        sizes = [int((np.abs(size - norm[i]) < INDEX_SIGMA_R*sig_r[i]).sum())
                 for i in order]
        if sum(sizes) >= 150:
            break
    probe = np.argsort(-counts)[:80]
    q_probe = q[probe]

    def judged(matrix):
        """(-count on the probe, capped score on the probe): lower is better."""
        found, _, _ = count_indexed(q_probe, matrix)
        return (-found, robust_score(q_probe, matrix))

    best = ((0, np.inf), None, 0)
    trials = 0
    for a in range(len(order)):
        i = order[a]
        near_i = np.flatnonzero(np.abs(size - norm[i]) < INDEX_SIGMA_R*sig_r[i])
        if not len(near_i):
            continue
        for b in range(a + 1, len(order)):
            j = order[b]
            near_j = np.flatnonzero(np.abs(size - norm[j])
                                    < INDEX_SIGMA_R*sig_r[j])
            if not len(near_j):
                continue
            want = float(np.dot(q[i], q[j])/(norm[i]*norm[j]))
            for p in near_i:
                cosines = (vectors[near_j] @ vectors[p])/(size[near_j]*size[p])
                gap = np.abs(cosines - want)
                match = near_j[np.argsort(gap)][:20]
                match = [m for m, g in zip(match, np.sort(gap)[:20])
                         if g < INDEX_ANGLE_TOL]
                for sec in match:
                    if trials >= budget:
                        break
                    trials += 1
                    spin = _two_vector_rotation(vectors[p], vectors[sec],
                                                q[i], q[j])
                    if spin is None:
                        continue
                    matrix = spin @ cell_to_basis(cell)
                    value = judged(matrix)
                    if value < best[0]:
                        count, _, _ = count_indexed(q, matrix)
                        best = (value, matrix, count)
                if trials >= budget:
                    break
            if trials >= budget:
                break
        if trials >= budget:
            break
    if best[1] is None:
        say("   %-14s no orientation puts that cell on these peaks (%d trials)"
            % (tag, trials))
        return None
    matrix = best[1]
    for _ in range(INDEX_ROUNDS):
        score, hkl, good = count_indexed(q, matrix)
        better = refine_rotation(q, hkl, cell, good)
        if better is None:
            break
        if judged(better) <= judged(matrix):
            matrix = better
        else:
            break
    score, hkl, good = count_indexed(q, matrix)
    chance = _chance(q, matrix)
    radial, across = weighted_residual(q[good], hkl[good], matrix)
    state.update(basis=matrix, hkl=hkl, good=good, free_cell=list(cell),
                 ub=matrix)
    say("   %-14s %3d of %3d peaks, %.1f by chance (%4.0f x), score %.3f,"
        % (tag, score, len(q), chance, score/max(chance, 1e-9),
           robust_score(q, matrix)))
    say("   %-14s %d trials, %d refinement rounds; residuals %.2f sigma along"
        % ("", trials, INDEX_ROUNDS, float(np.sqrt(np.mean(radial**2)))))
    say("   %-14s Q, %.2f across" % ("", float(np.sqrt(np.mean(across**2)))))
    if score/max(chance, 1e-9) < 5.0:
        say("   %-14s WARNING: barely above chance.  These assignments are not"
            % "")
        say("   %-14s worth refining a cell on." % "")
    return matrix


def _two_vector_rotation(g1, g2, q1, q2):
    """The rotation taking the pair (g1, g2) as close as it can to (q1, q2)."""
    def frame(u, v):
        e1 = u/np.linalg.norm(u)
        w = v - (v @ e1)*e1
        if np.linalg.norm(w) < 1e-9:
            return None
        e2 = w/np.linalg.norm(w)
        return np.array([e1, e2, np.cross(e1, e2)]).T
    left, right = frame(q1, q2), frame(g1, g2)
    if left is None or right is None:
        return None
    return left @ right.T


def _too_dense(matrix, q_max, observed):
    """Reject a lattice with far more nodes than there are peaks.

    This is what let a 100 A cell through: its nodes are so close together that
    almost any peak sits near one, and the count of indexed peaks stays high
    while meaning nothing.
    """
    nodes = 4.0/3.0*np.pi*q_max**3/abs(np.linalg.det(matrix))
    return nodes > MAX_NODES_PER_PEAK*max(observed, 1)


def _lattice_of(tag, state):
    q, counts = state["q"], state["counts"]
    if len(q) < 12:
        say("   %-14s only %d peaks - not enough to find a lattice"
            % (tag, len(q)))
        return None
    q_max = float(np.linalg.norm(q, axis=1).max())
    order = np.argsort(counts)[::-1][:INDEX_PEAKS]
    strong = q[order]
    pool = [strong[i] for i in range(len(strong))]
    for i in range(len(strong)):
        for j in range(i + 1, len(strong)):
            pool.append(strong[i] - strong[j])
    pool = [v for v in pool if np.linalg.norm(v) > 0.15]
    pool.sort(key=lambda v: np.linalg.norm(v))
    unique = []
    for v in pool:
        if all(np.linalg.norm(v - u) > 0.05 and np.linalg.norm(v + u) > 0.05
               for u in unique):
            unique.append(v)
        if len(unique) >= 30:
            break
    best = (-np.inf, None)
    refused = 0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            for k in range(j + 1, len(unique)):
                matrix = np.array([unique[i], unique[j], unique[k]]).T
                if abs(np.linalg.det(matrix)) < 1e-3:
                    continue
                if _too_dense(matrix, q_max, len(q)):
                    refused += 1
                    continue
                score, _, _ = count_indexed(q, matrix)
                excess = score - 3.0*_chance(q, matrix)
                if excess > best[0]:
                    best = (excess, matrix)
    if refused:
        say("   %-14s %d triples refused for having more than %d lattice nodes"
            % (tag, refused, MAX_NODES_PER_PEAK))
        say("   %-14s per observed peak - a lattice that dense indexes anything"
            % "")
    if best[1] is None or best[0] < 8:
        say("   %-14s no lattice stands out above chance; look at steps 4 and 5"
            % tag)
        return None
    matrix = reduce_basis(best[1])
    score, hkl, good = count_indexed(q, matrix)
    cell = basis_to_cell(matrix)
    chance = _chance(q, matrix)
    state.update(basis=matrix, hkl=hkl, good=good, free_cell=cell)
    say("   %-14s %3d of %3d peaks, %.1f by chance, %5.0f x chance"
        % (tag, score, len(q), chance, score/max(chance, 1e-9)))
    say("   %-14s free cell %.4f %.4f %.4f  %.2f %.2f %.2f"
        % ("", cell[0], cell[1], cell[2], cell[3], cell[4], cell[5]))
    return cell


def step_index():
    head("6. %s" % ("put the given cell on the peaks of each run" if CELL
                    else "find the lattice of each run - nothing assumed"))
    if "sigma_radial" not in SHARED:
        if not need(step_resolution, "there is no resolution function"):
            return None
    if CELL:
        say("   cell given: a=%.4f b=%.4f c=%.4f alpha=%.3f beta=%.3f"
            " gamma=%.3f" % tuple(CELL))
        say("   so the only unknown is the orientation of each run.  Two peaks")
        say("   and two reflections fix a rotation, so pairs are matched on")
        say("   both lengths within %.1f radial sigma and the angle to %.3f in"
            % (INDEX_SIGMA_R, INDEX_ANGLE_TOL))
        say("   cosine, seeds taken at |Q| < %.1f where a length still means"
            % INDEX_SEED_QMAX)
        say("   something.  A candidate is judged, as in sx_magic, by how many")
        say("   of the 80 brightest peaks it indexes (the cell is given, so a")
        say("   denser lattice cannot buy a count), the capped Mahalanobis sum")
        say("   breaking ties.  Then %d rounds of re-indexing and re-fitting the"
            % INDEX_ROUNDS)
        say("   rotation with the cell held.")
        if SYSTEM_GIVEN:
            say("   --system is ignored: a cell already fixes what a system")
            say("   would only constrain.")
        say()
        say("   %-14s %s" % ("run", "indexed"))
        for tag, _, state in each_run():
            if state.get("q") is not None:
                _orientation(tag, state, CELL)
        return [CELL]
    say("   the measured Q of a Bragg peak IS a reciprocal lattice vector, so")
    say("   the job is to find the lattice holding the brightest ones.  The")
    say("   candidates are those vectors and their differences; every")
    say("   non-coplanar triple of the 30 shortest is tried, and an index")
    say("   counts as integer when the node is the nearest BY DIRECTION among")
    say("   all 27 neighbours, with |Q| allowed to be wrong by %.1f radial"
        % INDEX_SIGMA_R)
    say("   sigma and the transverse offset inside %.1f sigma -" % INDEX_SIGMA_T)
    say("   anisotropic, which is the part that makes it work here.  A triple")
    say("   whose lattice would have more than %d nodes per observed peak is"
        % MAX_NODES_PER_PEAK)
    say("   refused: such a lattice indexes anything and means nothing.")
    say()
    say("   %-14s %s" % ("run", "indexed"))
    for tag, _, state in each_run():
        if state.get("q") is not None:
            _lattice_of(tag, state)
    return [state.get("free_cell") for _, _, state in each_run()]


# ---------------------------------------------------- one cell, many runs --
def pattern_of(cell, tol=1e-4):
    """Which cell parameters are free, given the equalities the cell already has.

    A cell handed in carries its own symmetry: if a equals b, or an angle is 90
    or 120, that is a statement, and refining it away would be refining out the
    symmetry the operator asserted.  So the pattern of the given cell is kept
    and only what it leaves open is refined.  This is why --system is not used
    when a cell is given: the two would say the same thing twice, and disagree.
    """
    a, b, c, al, be, ga = cell
    lengths = [a, b, c]
    groups = []
    for i in range(3):
        for g in groups:
            if abs(lengths[i] - lengths[g[0]]) <= tol*max(lengths[i], 1.0):
                g.append(i)
                break
        else:
            groups.append([i])
    angles = [al, be, ga]
    fixed, free = {}, []
    for i in range(3):
        if abs(angles[i] - 90.0) < 1e-3 or abs(angles[i] - 120.0) < 1e-3:
            fixed[i] = angles[i]
        else:
            for g in free:
                if abs(angles[i] - angles[g[0]]) < 1e-3:
                    g.append(i)
                    break
            else:
                free.append([i])
    return groups, fixed, free


def pattern_pack(cell, pattern):
    groups, _, free = pattern
    lengths = [cell[0], cell[1], cell[2]]
    angles = [cell[3], cell[4], cell[5]]
    return ([float(np.mean([lengths[i] for i in g])) for g in groups]
            + [float(np.mean([angles[i] for i in g])) for g in free])


def pattern_unpack(values, pattern):
    groups, fixed, free = pattern
    lengths = [0.0, 0.0, 0.0]
    angles = [0.0, 0.0, 0.0]
    for value, g in zip(values[:len(groups)], groups):
        for i in g:
            lengths[i] = float(value)
    for index, fixed_value in fixed.items():
        angles[index] = float(fixed_value)
    for value, g in zip(values[len(groups):], free):
        for i in g:
            angles[i] = float(value)
    return lengths + angles


def pattern_words(pattern):
    groups, fixed, free = pattern
    names = "abc"
    parts = ["=".join(names[i] for i in g) for g in groups]
    angle_names = ["alpha", "beta", "gamma"]
    parts += ["=".join(angle_names[i] for i in g) for g in free]
    held = ["%s=%.0f" % (angle_names[i], v) for i, v in sorted(fixed.items())]
    return ", ".join(parts) + (" ; held: %s" % ", ".join(held) if held else "")


CENTRING_NAMES = {"P": "Primitive", "A": "A-face centred",
                  "B": "B-face centred", "C": "C-face centred",
                  "I": "Body centred", "F": "All-face centred",
                  "ROBV": "Rhombohedrally centred obverse",
                  "RREV": "Rhombohedrally centred reverse"}
CELL_NAMES = ["a", "b", "c", "alpha", "beta", "gamma"]


def allowed_by_centring(hkl, centring):
    """True where the centring does not forbid the reflection."""
    h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    rule = (centring or "P").upper()
    if rule == "A":
        return (k + l) % 2 == 0
    if rule == "B":
        return (h + l) % 2 == 0
    if rule == "C":
        return (h + k) % 2 == 0
    if rule == "I":
        return (h + k + l) % 2 == 0
    if rule == "F":
        return ((h + k) % 2 == 0) & ((h + l) % 2 == 0) & ((k + l) % 2 == 0)
    if rule == "ROBV":
        return (-h + k + l) % 3 == 0
    if rule == "RREV":
        return (h - k + l) % 3 == 0
    return np.ones(len(hkl), dtype=bool)


def extinction_report(hkl):
    """Which centring do the indexed reflections themselves say this is?

    A lattice centring forbids a whole class of reflections, so the class has to
    come out empty.  This costs nothing, needs no intensities and no known
    structure, and it is the strongest check in the workflow: if the class that
    should be empty holds its share of the peaks, the indices are wrong and
    nothing downstream means anything.
    """
    if not len(hkl):
        return None
    say()
    say("   what the indices themselves say about the centring - a forbidden")
    say("   class has to come out empty:")
    best = ("P", 0.0)
    for rule in ("P", "A", "B", "C", "I", "F"):
        allowed = allowed_by_centring(hkl, rule)
        share = 1.0 - float(allowed.mean())
        if rule == "P":
            say("     %-4s every reflection allowed" % rule)
            continue
        expected = {"A": 0.5, "B": 0.5, "C": 0.5, "I": 0.5, "F": 0.75}[rule]
        say("     %-4s %5.1f %% of the indexed peaks fall in the class it"
            " forbids (a random lattice gives %.0f %%)"
            % (rule, 100*share, 100*expected))
        if share < 0.05 and expected - share > best[1]:
            best = (rule, expected - share)
    if best[0] != "P":
        say("   the data look %s-centred: that class is empty." % best[0])
    else:
        say("   no class is empty, so either the lattice is primitive or the")
        say("   indexing is wrong.  If the crystal is centred, this is the")
        say("   place the run has failed, whatever the other numbers say.")
    return best[0]


def knobs_of(cell, free_names):
    """Groups of cell parameters that move together, and whether they move.

    Two constraints meet here.  The cell as given carries its own symmetry, so
    parameters equal in it stay equal (pattern_of).  And --refine may name a
    subset: --refine a,b means c and the angles are held wherever the pattern
    leaves them free.
    """
    groups, fixed, free = pattern_of(cell)
    out = []
    for g in groups:
        index = list(g)
        out.append((index, free_names is None
                    or any(CELL_NAMES[i] in free_names for i in index)))
    for g in free:
        index = [3 + i for i in g]
        out.append((index, free_names is None
                    or any(CELL_NAMES[j] in free_names for j in index)))
    return out, fixed


def knob_pack(cell, knobs):
    return [float(np.mean([cell[i] for i in index]))
            for index, moving in knobs[0] if moving]


def knob_unpack(values, cell, knobs):
    out = list(cell)
    step = iter(values)
    for index, moving in knobs[0]:
        value = next(step) if moving else float(np.mean([cell[i]
                                                        for i in index]))
        for i in index:
            out[i] = float(value)
    for i, held in knobs[1].items():
        out[3 + i] = float(held)
    return out


def knob_words(knobs):
    moving = [",".join(CELL_NAMES[i] for i in index)
              for index, flag in knobs[0] if flag]
    held = [",".join(CELL_NAMES[i] for i in index)
            for index, flag in knobs[0] if not flag]
    held += ["%s=%.0f" % (CELL_NAMES[3 + i], v)
             for i, v in sorted(knobs[1].items())]
    return ("free: %s" % (", ".join(moving) or "nothing")
            + ("  held: %s" % ", ".join(held) if held else ""))


def show_matrices(have, cell):
    """UB, U and B, per run, in the convention Q = 2 pi UB h."""
    basis = cell_to_basis(cell)/(2*np.pi)
    say()
    say("   B, from the cell alone (rows are the reciprocal axes, 1/A):")
    for row in basis:
        say("     %10.6f %10.6f %10.6f" % tuple(row))
    spins = {}
    for tag, state in have:
        ub = state["ub"]/(2*np.pi)
        u = ub @ np.linalg.inv(basis)
        spins[tag] = u
        axis = spin_of(u)
        angle = float(np.degrees(np.linalg.norm(axis)))
        say()
        say("   %s" % tag)
        say("     UB (1/A)                          U (orthogonal)")
        for i in range(3):
            say("     %9.5f %9.5f %9.5f      %8.5f %8.5f %8.5f"
                % (ub[i, 0], ub[i, 1], ub[i, 2], u[i, 0], u[i, 1], u[i, 2]))
        error = float(np.abs(u @ u.T - np.eye(3)).max())
        say("     U is orthogonal to %.1e, det %+.6f; it is a rotation of"
            % (error, float(np.linalg.det(u))))
        say("     %.3f deg about (%.4f %.4f %.4f) in the sample frame"
            % ((angle,) + tuple(axis/max(np.linalg.norm(axis), 1e-12))))
    if len(have) > 1:
        say()
        say("   and the orientations of the runs against each other.  With a")
        say("   goniometer log they must coincide; without one, the rotation")
        say("   between them MEASURES the axis the sample turned about:")
        tags = [tag for tag, _ in have]
        ops = lattice_rotations(basis)
        say("     (an orientation is only defined up to the %d rotations of the"
            % len(ops))
        say("      lattice itself, so for each run the equivalent that makes")
        say("      the axes agree best is taken)")
        first = tags[0]
        # With the goniometer applied from a log, every run should come out
        # with the SAME orientation in the sample frame.  Check that first:
        # the rotation from run 1 to run k, modulo the lattice's own rotations.
        apart = [min(float(np.degrees(np.linalg.norm(
                     spin_of(spins[t] @ op @ spins[first].T))))
                     for op in ops) for t in tags[1:]]
        if apart and max(apart) < 1.0:
            say("     every run has the same orientation in the sample frame, "
                "to %.3f deg" % max(apart))
            say("     (%s): the goniometer angles in the files are right."
                % ", ".join("%s %.3f" % (t, d) for t, d in zip(tags[1:], apart)))
            return
        choices = [[spins[t] @ op for op in ops] for t in tags[1:]]
        best = None
        for combo in itertools.product(*[range(len(c)) for c in choices]):
            vectors = [spin_of(choices[k][i] @ spins[first].T)
                       for k, i in enumerate(combo)]
            units = [v/max(np.linalg.norm(v), 1e-12) for v in vectors]
            spread = max([float(np.degrees(np.arccos(np.clip(
                abs(units[0] @ u), -1, 1)))) for u in units[1:]] or [0.0])
            # prefer the smallest turns among equally good axes
            key = (round(spread, 3), sum(np.linalg.norm(v) for v in vectors))
            if best is None or key < best[0]:
                best = (key, vectors, units, spread)
        _, vectors, units, spread = best
        for tag, vector, unit in zip(tags[1:], vectors, units):
            angle = float(np.degrees(np.linalg.norm(vector)))
            say("     %-10s -> %-10s %8.3f deg about (%+.4f %+.4f %+.4f)"
                % (first, tag, angle, unit[0], unit[1], unit[2]))
        if len(units) > 1:
            say("     the axes agree to %.2f deg.  %s" % (spread,
                "One axis for all of them, as a goniometer must give."
                if spread < 3.0 else
                "They do NOT agree, so at least one orientation is wrong."))


def lattice_rotations(basis):
    """Proper rotations that map the lattice onto itself, as matrices acting
    on U from the right: U and U @ R describe the same crystal.

    Found by brute force over integer matrices with entries -1, 0, 1 that keep
    the metric; that covers every Bravais lattice in a reduced setting.
    """
    inverse = np.linalg.inv(basis)
    found = []
    for entries in itertools.product((-1, 0, 1), repeat=9):
        whole = np.array(entries, dtype=float).reshape(3, 3)
        if abs(np.linalg.det(whole) - 1.0) > 1e-6:
            continue
        rot = basis @ whole @ inverse
        if np.abs(rot @ rot.T - np.eye(3)).max() < 1e-4:
            found.append(rot)
    return found or [np.eye(3)]


def _residual_now(have):
    """Rms residual in sigmas of whatever basis each run currently carries."""
    pieces = []
    for _, state in have:
        good = state["good"]
        radial, across = weighted_residual(state["q"][good],
                                           state["hkl"][good], state["basis"])
        pieces.append(np.concatenate([radial, across]))
    every = np.concatenate(pieces) if pieces else np.zeros(1)
    return float(np.sqrt(np.mean(every**2)))


def step_cell():
    head("7. %s" % {"none": "no refinement - the cell and orientations stand",
                    "orientation": "refine the orientations, hold the cell",
                    "cell": "refine the cell and the orientations"}[REFINE])
    have = [(tag, state) for tag, _, state in each_run()
            if state.get("basis") is not None]
    if not have:
        if need(step_index, "no run has a lattice"):
            have = [(tag, state) for tag, _, state in each_run()
                    if state.get("basis") is not None]
    if not have:
        say("   no run has a lattice")
        return None
    reflections = sum(int(state["good"].sum()) for _, state in have)
    say("   %d runs, %d indexed reflections in all" % (len(have), reflections))
    say("   rms residual as it stands: %.3f sigma" % _residual_now(have))

    if REFINE == "none":
        SHARED["cell"] = list(CELL) if CELL else list(have[0][1]["free_cell"])
        for _, state in have:
            state["ub"] = state["basis"]
        _write_ub(have)
        say("   nothing is refined, by request.  The cell is %s"
            % ("the one given" if CELL else "the one the search found"))
        return SHARED["cell"]
    try:
        from scipy.optimize import least_squares
    except Exception:
        say("   scipy is not here, and the refinement needs it")
        return None
    q_of = {tag: state["q"][state["good"]] for tag, state in have}
    h_of = {tag: state["hkl"][state["good"]] for tag, state in have}

    if CELL:
        pattern = knobs_of(CELL, REFINE_NAMES)
        start_cell = list(CELL)
        systems = [(None, pattern)]
        say("   %s" % knob_words(pattern))
        say("   the cell keeps the equalities it was given in, so a=b stays")
        say("   a=b and an angle of 90 or 120 is held; --refine names which of")
        say("   the rest may move.")
    else:
        start_cell = list(np.mean([state["free_cell"] for _, state in have],
                                 axis=0))
        if len(have) > 1:
            spread = np.std([state["free_cell"] for _, state in have], axis=0)
            say("   the free cells of the runs agree to %.4f %.4f %.4f A and"
                % tuple(spread[:3]))
            say("   %.2f %.2f %.2f deg before anything is constrained"
                % tuple(spread[3:]))
        systems = [("triclinic", None), (SYSTEM, None)]

    def make(values, mode, pattern, free_cell):
        if free_cell:
            n = (len(knob_pack(start_cell, pattern)) if pattern
                 else FREE[mode])
            cell = (knob_unpack(values[:n], start_cell, pattern) if pattern
                    else unpack(values[:n], mode))
        else:
            n = 0
            cell = list(start_cell)
        basis = cell_to_basis(cell)
        out = {}
        for index, (tag, _) in enumerate(have):
            out[tag] = rotation(values[n + 3*index: n + 3*index + 3]) @ basis
        return cell, out

    def residual(values, mode, pattern, free_cell):
        _, matrices = make(values, mode, pattern, free_cell)
        pieces = []
        for tag, _ in have:
            radial, across = weighted_residual(q_of[tag], h_of[tag],
                                               matrices[tag])
            pieces.append(radial)
            pieces.append(across)
        return np.concatenate(pieces)

    free_cell = REFINE == "cell"
    results = {}
    for mode, pattern in systems:
        label = mode or "as given"
        cell = list(start_cell)
        guess = []
        if free_cell:
            guess += list(knob_pack(cell, pattern) if pattern
                          else pack(cell, mode))
        base = cell_to_basis(knob_unpack(guess, cell, pattern)
                             if (free_cell and pattern)
                             else (unpack(guess, mode) if free_cell else cell))
        for _, state in have:
            guess += list(spin_of(state["basis"] @ np.linalg.inv(base)))
        try:
            out = least_squares(residual, np.array(guess, dtype=float),
                                args=(mode, pattern, free_cell), max_nfev=8000)
        except Exception as exc:
            say("   refinement failed for %s: %s" % (label, exc))
            continue
        cost = float(np.sqrt(np.mean(out.fun**2)))
        final, matrices = make(out.x, mode, pattern, free_cell)
        results[label] = (cost, final, matrices)
        say("   %-13s rms %.3f sigma   a=%.4f b=%.4f c=%.4f"
            % (label, cost, final[0], final[1], final[2]))
        say("   %-13s                  alpha=%.3f beta=%.3f gamma=%.3f"
            % ("", final[3], final[4], final[5]))
    if not results:
        return None
    if CELL:
        label = "as given"
        if free_cell:
            change = [100*(results[label][1][i]/CELL[i] - 1) for i in range(3)]
            say()
            say("   the edges moved %+.3f %+.3f %+.3f %% from the cell given"
                % tuple(change))
            common = float(np.mean(change))
            spread = float(np.max(change) - np.min(change))
            if spread < 0.3 and abs(common) > 0.1:
                say("   they moved together, by %+.2f %% on average, and a"
                    % common)
                say("   common shift is a wavelength-calibration matter, not")
                say("   the crystal: t0 and L enter only as a pair.")
            elif spread >= 0.3:
                say("   they did NOT move together - the spread is %.2f %% -"
                    % spread)
                say("   and t0 cannot do that.  An error proportional to")
                say("   1/lambda distorts the shape rather than scaling it,")
                say("   and a shape change of this size means the index")
                say("   assignments are wrong, not the calibration.")
    else:
        if "triclinic" in results and SYSTEM in results and SYSTEM != "triclinic":
            free, held = results["triclinic"][0], results[SYSTEM][0]
            say()
            say("   %s removes %d free parameters and costs %.3f sigma in rms"
                % (SYSTEM, 6 - FREE[SYSTEM], held - free))
            say("   (%.3f -> %.3f).  %s"
                % (free, held, "The data support it."
                   if held < free + 0.15 else
                   "The data DO NOT support it - the free cell fits markedly "
                   "better, so the system is wrong or there is a second grain."))
        label = SYSTEM if SYSTEM in results else "triclinic"
    cost, cell, matrices = results[label]
    SHARED["cell"] = cell
    say()
    say("   re-indexing on the refined cell, because the assignments were made")
    say("   with the old one:")
    for tag, state in have:
        state["ub"] = matrices[tag]
        was = int(state["good"].sum())
        score, hkl, good = count_indexed(state["q"], matrices[tag])
        chance = _chance(state["q"], matrices[tag])
        changed = int((hkl != state["hkl"]).any(axis=1).sum())
        state["hkl"], state["good"] = hkl, good
        say("     %-12s %3d -> %3d indexed (%4.0f x chance), %d assignments"
            % (tag, was, score, score/max(chance, 1e-9), changed))
        say("     %-12s changed" % "")
    _write_ub(have)
    show_matrices(have, cell)
    every = np.vstack([state["hkl"][state["good"]] for _, state in have])
    found = extinction_report(every)
    if found and found != "P" and (CENTRING or "P").upper() == "P":
        say()
        say("   NOTE: --centring is P but the indices look %s-centred.  Pass"
            % found)
        say("   --centring %s so the prediction and the integration do not"
            % found)
        say("   spend half their work on reflections that cannot exist.")
    if (CENTRING or "P").upper() != "P":
        allowed = allowed_by_centring(every, CENTRING)
        share = 1.0 - float(allowed.mean())
        say()
        say("   with --centring %s, %.1f %% of the indexed peaks are in the"
            % (CENTRING, 100*share))
        say("   forbidden class.  %s"
            % ("That is consistent." if share < 0.05 else
               "THE INDEXING HAS FAILED: that class must be empty, and it is "
               "not.  Nothing downstream means anything until it is."))
    return cell


def _write_ub(have):
    for tag, state in have:
        if state.get("peaks") and state["peaks"] in mtd:
            flat = [float(v) for v in (state["ub"]/(2*np.pi)).ravel()]
            call("SetUB", Workspace=state["peaks"], UB=flat)
    say("   UB written onto every peaks workspace; the cell is shared, the")
    say("   orientation is each run's own")


# --------------------------------------------------------- normalisation --
def step_normalise():
    head("8. normalisation, one file for every run")
    if not NORMALISATION:
        say("   none given, so intensities come out in counts: comparing one")
        say("   reflection with another at a different wavelength, or in a")
        say("   different part of the bank, is then not meaningful.")
        return None
    if not os.path.exists(NORMALISATION):
        say("   %s is not here" % NORMALISATION)
        return None
    if "lam" not in SHARED and not need(step_propose, "no band was proposed"):
        return None
    lam = SHARED["lam"]
    first = RUN[tag_of(RUNS[0])]
    name = "%s_norm" % PREFIX
    try:
        LoadEventNexus(Filename=NORMALISATION, OutputWorkspace=name,
                       LoadMonitors=False)
    except (RuntimeError, ValueError) as trouble:
        say("   %s cannot be read as an event NeXus file: %s"
            % (NORMALISATION, str(trouble).splitlines()[-1]))
        say("   A raw McStas .h5 has to go through the converter first, like")
        say("   the sample runs.  Going on WITHOUT normalisation.")
        return None
    if IDF:
        LoadInstrument(Workspace=name, Filename=IDF, RewriteSpectraMap=False)
    unwrap_frames("normalisation", name)
    if mtd[name].getNumberHistograms() != len(first["two_theta"]):
        say("   %s has %d spectra and the runs have %d - not the same detector"
            % (NORMALISATION, mtd[name].getNumberHistograms(),
               len(first["two_theta"])))
        return None
    if T0:
        ChangeBinOffset(InputWorkspace=name, OutputWorkspace=name,
                        Offset=-float(T0))
    ConvertUnits(InputWorkspace=name, OutputWorkspace=name,
                 Target="Wavelength", EMode="Elastic")
    CropWorkspace(InputWorkspace=name, OutputWorkspace=name, XMin=lam[0],
                  XMax=lam[1])
    Rebin(InputWorkspace=name, OutputWorkspace=name,
          Params="%f,%f,%f" % (lam[0], (lam[1] - lam[0])/NORM_LAMBDA_BINS,
                               lam[1]), PreserveEvents=False)
    counts = np.array(mtd[name].extractY())
    total = float(counts.sum())
    two_theta, far = first["two_theta"], first["l2"]
    live = counts.sum(axis=1) > 0
    say("   %.0f events over %d voxels, %d of which hold anything (%.1f %%)"
        % (total, len(live), int(live.sum()), 100.0*live.mean()))
    if live.mean() < 0.9:
        say("   the empty ones are left out of every average: on a detector")
        say("   that places an event at the face shared by two voxels, half of")
        say("   them never receive anything, and counting them as live")
        say("   detectors reading zero would bias every background.")
    per_cell = total/NORM_LAMBDA_BINS
    groups = int(max(1, min(4096, per_cell*NORM_TARGET**2)))
    side = max(1, int(np.sqrt(groups)))
    say("   %.0f events per wavelength bin, so %.0f %% per cell allows about"
        % (per_cell, 100*NORM_TARGET))
    say("   %d groups: %d bands of 2theta by %d of L2." % (side*side, side, side))
    theta_edges = np.quantile(two_theta[live], np.linspace(0, 1, side + 1))
    far_edges = np.quantile(far[live], np.linspace(0, 1, side + 1))
    theta_band = np.clip(np.searchsorted(theta_edges, two_theta, "right") - 1,
                         0, side - 1)
    far_band = np.clip(np.searchsorted(far_edges, far, "right") - 1, 0, side - 1)
    group = theta_band*side + far_band
    table = np.zeros((side*side, NORM_LAMBDA_BINS))
    held = np.zeros(side*side)
    for slot in range(side*side):
        pick = live & (group == slot)
        if pick.any():
            # per VOXEL, not per group (section 114): the sample intensity is
            # what one voxel's worth of peak collected, and a group sum grows
            # with the number of voxels the group happens to hold
            table[slot] = counts[pick].sum(axis=0)/pick.sum()
            held[slot] = pick.sum()
    edges = np.array(mtd[name].readX(0), dtype=float)
    filled = table[held > 0]*held[held > 0, None]     # counts per cell
    reached = float(np.mean(filled > 1.0/NORM_TARGET**2)) if len(filled) else 0.0
    say("   the table is empirical - no model in it.  It carries the spectrum,")
    say("   the transmission of the detector material in front of a voxel and")
    say("   the voxel efficiency together, and %.0f %% of its cells reach the"
        % (100*reached))
    say("   wanted %.1f %% counting error." % (100*NORM_TARGET))
    say("   L2 bands matter as much as 2theta bands: on a deep detector the")
    say("   material in front of a voxel absorbs as lambda, so a far voxel")
    say("   sees a different spectrum, not just fewer counts.")
    centres = 0.5*(edges[1:] + edges[:-1])
    good = held > 0
    # one constant for every run (section 116): the old factor was divided by
    # the mean over the reflections of each run, which gave every run its own
    # arbitrary scale
    per_voxel_mean = float(table[good].mean()) if good.any() else 1.0
    SHARED["norm"] = dict(table=table/max(per_voxel_mean, 1e-12), held=held,
                          lam=centres, group=group)
    try:
        SHARED["norm"]["depth"] = vanadium_model(name, counts, centres)
    except Exception as trouble:
        say("   the depth model could not be built (%s); the groups table is"
            % str(trouble).splitlines()[-1])
        say("   used instead")
    return SHARED["norm"]


def _weighted_additive(value, weight, rounds=80):
    """least squares fit of value ~ row + column, with weights"""
    row = np.zeros(value.shape[0])
    col = np.zeros(value.shape[1])
    for _ in range(rounds):
        col = ((weight*(value - row[:, None])).sum(0)
               / np.maximum(weight.sum(0), 1e-9))
        row = ((weight*(value - col[None, :])).sum(1)
               / np.maximum(weight.sum(1), 1e-9))
    return row, col


def vanadium_model(name, counts, lam):
    """sx_magic's normalisation (its build_vanadium, plan sections 75-79).

      Phi(lambda)            the incident spectrum;
      exp(-k(depth) lambda)  transmission of the detector in front of a voxel,
                             32 numbers, worth a factor ~10 from front to back;
      epsilon(voxel)         solid angle x depth scale x share along the row -
                             NOT the counts of the voxel itself, which at ~20
                             weighted events per voxel carry 60 % noise.
    Everything is normalised by constants, the same for every run.
    """
    along_n, rows_n, levels = BANK_A_SHAPE
    spectra = counts.shape[0]
    ws = mtd[name]
    ids = np.full(spectra, -1, dtype=np.int64)
    for i in range(spectra):
        got = ws.getSpectrum(i).getDetectorIDs()
        if got:
            ids[i] = int(list(got)[0])
    bank_a = (ids >= 1) & (ids <= along_n*rows_n*levels)
    local = np.where(bank_a, ids - 1, 0)
    along = local % along_n
    depth = local // (along_n*rows_n)
    total = counts.sum(axis=1)
    live = bank_a & (total > 0)
    table = np.array([counts[live & (depth == level)].sum(axis=0)
                      for level in range(levels)])
    usable = table > 20
    value = np.log(np.where(usable, np.maximum(table, 1.0), 1.0))
    weight = np.where(usable, table, 0.0)
    k = np.zeros(levels)
    for _ in range(150):
        level_scale, phi = _weighted_additive(value + k[:, None]*lam[None, :],
                                             weight, rounds=10)
        left = value - level_scale[:, None] - phi[None, :]
        for level in range(levels):
            k[level] = -((weight[level]*left[level]*lam).sum()
                         / max((weight[level]*lam*lam).sum(), 1e-9))
    k = k - k[0]
    level_scale, phi = _weighted_additive(value + k[:, None]*lam[None, :],
                                         weight, rounds=60)
    flux = np.exp(phi)
    flux = flux/np.max(flux)
    call("SolidAngle", InputWorkspace=name, OutputWorkspace=name + "_sa")
    solid = np.array(mtd[name + "_sa"].extractY()).ravel()
    DeleteWorkspace(name + "_sa")
    solid = np.where(np.isfinite(solid) & (solid > 0), solid, np.nan)
    solid = solid/np.nanmedian(solid[live])
    depth_scale = np.ones(levels)
    for level in range(levels):
        here = live & (depth == level) & np.isfinite(solid)
        if here.sum() > 100:
            depth_scale[level] = float(np.mean(total[here]/solid[here]))
    depth_scale = depth_scale/depth_scale[0]
    expect = solid*depth_scale[depth]
    share = np.ones(along_n)
    for slot in range(along_n):
        here = live & (along == slot) & np.isfinite(expect) & (expect > 0)
        if here.sum() > 100:
            share[slot] = float(np.mean(total[here]/expect[here]))
    share = share/np.mean(share)
    efficiency = np.where(np.isfinite(solid) & live,
                          solid*depth_scale[depth]*share[along], np.nan)
    say("   depth model (as sx_magic): Phi(lambda) varies %.1fx across the band,"
        % float(flux.max()/max(flux.min(), 1e-12)))
    say("   exp(-k lambda) of the back shell at 1 A is %.3f of the front one,"
        % float(np.exp(-k[-1])))
    say("   depth scale back/front %.3f" % float(depth_scale[-1]))
    return dict(lam=lam, flux=flux, k=k, depth=depth, efficiency=efficiency,
                bank_a=bank_a)


def normalisation_factor(workspace_index, wavelength, model=None):
    """What a reflection is divided by; the same scale for every run."""
    table = SHARED.get("norm")
    if not table:
        return np.ones(len(wavelength))
    model = NORM_MODEL if model is None else model
    if model == "depth" and table.get("depth"):
        m = table["depth"]
        index = np.clip(np.asarray(workspace_index, dtype=int), 0,
                        len(m["depth"]) - 1)
        good = m["flux"] > 0
        spectrum = np.interp(wavelength, m["lam"][good], m["flux"][good])
        eff = m["efficiency"][index]
        eff = np.where(np.isfinite(eff) & (eff > 0.05), eff, 1.0)
        through = np.exp(-m["k"][m["depth"][index]]*np.asarray(wavelength))
        return np.where(m["bank_a"][index], spectrum*eff*through, 1.0)
    slot = table["group"][np.clip(workspace_index, 0,
                                  len(table["group"]) - 1)]
    out = np.ones(len(wavelength))
    for i, (s, w) in enumerate(zip(slot, wavelength)):
        row = table["table"][s]
        if row.sum() > 0:
            out[i] = np.interp(w, table["lam"], row)
    return np.where(out > 0, out, 1.0)


# ------------------------------------------------------------- integrate --
# ----------------------------------------------- integration, from sx_magic.py
# Mantid's own ellipsoid was tried and did not work: IntegrateEllipsoids takes
# one PeakSize for all three axes, so a 5:1 shape cannot be expressed through
# it, and it fits the axis DIRECTIONS from a cloud that is mostly background.
# IntegratePeaksMD with Ellipsoid and FixQAxis did not apply the radii to the
# axes intended - swapping the order of the three changed nothing measurable.
#
# The second problem is worse and belongs to both: a background shell scaled
# from the peak radius reaches 4.5 sigma_r, which at |Q| = 4 is 0.38 A^-1
# against c* = 0.4995 - so what is subtracted as background is the NEXT Bragg
# peak, by an amount that grows with |Q|.
#
# So: bin the box in the frame attached to the reflection, sum inside the
# ellipsoid (r/a)^2 + (rho/b)^2 <= 1, and take the background from a collar
# displaced only ACROSS Q, where nothing else lives.  The grid is in units of
# the two semi-axes, so the masks are the same for every reflection.
LOCAL_HALF = (1.15, 2.0)        # box half-width, in units of a and of b
LOCAL_BINS = (47, 41, 41)
LOCAL_COLLAR = (1.0, 1.8)       # the background collar, in units of b
ERROR_SCALE = 1.0               # --error-scale: weighted events need about 2.5


def local_masks(collar=LOCAL_COLLAR):
    axes = [np.linspace(-h, h, n + 1)[:-1] + h/n
            for h, n in zip((LOCAL_HALF[0], LOCAL_HALF[1], LOCAL_HALF[1]),
                            LOCAL_BINS)]
    radial, first, second = np.meshgrid(*axes, indexing="ij")
    across = np.sqrt(first**2 + second**2)
    inside = radial**2 + across**2 <= 1.0
    shell = ((np.abs(radial) <= 1.0) & (across > collar[0])
             & (across <= collar[1]))
    return inside, shell


def isolated_peaks(peaks, n_sigma=None):
    """Which reflections do not share their integration volume with a neighbour.

    Two volumes with semi-axes (a, b, b) at separation D, decomposed along and
    across Q, keep clear when (D_r/2a)^2 + (D_t/2b)^2 >= 1.  Above |Q| = 4 the
    radial semi-axis exceeds half of c*, so it is the PEAKS that overlap, not
    just their background shells, and each then collects part of the other.
    """
    n_r = PEAK_SIGMAS_R if n_sigma is None else n_sigma
    n_t = PEAK_SIGMAS_T if n_sigma is None else n_sigma
    q = np.array([list(peak.getQSampleFrame()) for peak in peaks])
    if len(q) < 2:
        return np.ones(len(q), dtype=bool)
    modulus = np.linalg.norm(q, axis=1)
    keep = np.ones(len(q), dtype=bool)
    for index in range(len(q)):
        if modulus[index] <= 0.0:
            keep[index] = False
            continue
        a = n_r*float(sigma_of(SHARED["sigma_radial"], modulus[index]))
        b = n_t*float(sigma_of(SHARED["sigma_across"], modulus[index]))
        radial = q[index]/modulus[index]
        offset = q - q[index]
        along = offset @ radial
        across = np.sqrt(np.maximum((offset**2).sum(axis=1) - along**2, 0.0))
        touching = (along/(2.0*a))**2 + (across/(2.0*b))**2 < 1.0
        touching[index] = False
        keep[index] = not touching.any()
    return keep


def _fraction_inside(value, sigma, low, high):
    from math import erf, sqrt
    if sigma <= 0:
        return 1.0
    return max(0.0, 0.5*(erf((high - value)/(sigma*sqrt(2.0)))
                         - erf((low - value)/(sigma*sqrt(2.0)))))



# ------------------------------------------------ bank edges (section 112)
# A reflection whose peak voxel lies in the outermost segments or rows of bank
# A has part of its peak off the detector: with the calibrated vox chain these
# came out at 0.34 (outermost segment) and 0.61 (outermost rows) of |F|^2, and
# cutting them took R(F^2) from 0.189 to 0.121.  Distances in voxelization.py
# units: segments (a pair of voxels, 1.05 deg) from the two side edges, rows
# (anodes) from the top and the bottom.
EDGE_SEGMENTS = 2
EDGE_ROWS = 2


def near_bank_edge(detector_ids, segments=None, rows=None):
    """True where a detector id of bank A lies within the edge margin."""
    segments = EDGE_SEGMENTS if segments is None else segments
    rows = EDGE_ROWS if rows is None else rows
    ids = np.asarray(detector_ids, dtype=np.int64)
    n_vs_all, n_a_all, n_c_all = 120, 128, 32
    out = np.zeros(ids.shape, dtype=bool)
    bank_a = (ids >= 1) & (ids <= 2*60*128*32)
    local = ids[bank_a] - 1
    n_vs = local % n_vs_all
    n_a = (local // n_vs_all) % n_a_all
    segment = np.minimum(n_vs//2, (n_vs_all - 1 - n_vs)//2)
    row = np.minimum(n_a, n_a_all - 1 - n_a)
    out[bank_a] = (segment < segments) | (row < rows)
    return out

def _saturation(tag, label, nets, strong, q, fixed_text):
    """Smallest n holding SATURATION_KEEP of the plateau in every |Q| band."""
    plateau = nets[:, -1]
    fraction = nets/np.where(plateau != 0, plateau, np.nan)[:, None]
    picked = np.flatnonzero(strong)
    bands = max(1, len(picked)//SATURATION_BAND_MIN)
    edges = np.quantile(q[picked], np.linspace(0, 1, bands + 1))
    say("   %-14s %s (%s), median I(n)/I(%.1f) per band of |Q|:"
        % ("", label, fixed_text, SATURATION_N[-1]))
    say("   %-14s   |Q| A-1      n  " % "" + " ".join("%6.1f" % n
                                                     for n in SATURATION_N))
    chosen = SATURATION_N[0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        here = picked[(q[picked] >= lo) & (q[picked] <= hi)]
        level = np.nanmedian(fraction[here], axis=0)
        enough = [n for n, v in zip(SATURATION_N, level) if v >= SATURATION_KEEP]
        need = enough[0] if enough else SATURATION_N[-1]
        chosen = max(chosen, need)
        say("   %-14s   %4.1f-%4.1f  %3d  " % ("", lo, hi, len(here))
            + " ".join("%6.3f" % v for v in level) + "   -> %.1f" % need)
    return chosen


def choose_peak_sigmas(tag, table_n):
    """The ellipsoid at which strong peaks stop growing (sections 115, 117).

    table_n holds, per reflection, (net, var) on a grid n_r x n_t of
    SATURATION_N.  Strong = I/sigma > 10 at (3, 3).  First the size ACROSS Q
    with the radial axis at its largest, so the peak is not cut along Q while
    its width is measured; then the size ALONG Q at that n_t.  One ellipsoid
    for all runs: chosen on the first and kept.
    """
    global PEAK_SIGMAS_R, PEAK_SIGMAS_T
    ref = SATURATION_N.index(3.0)
    nets = np.array([row[0] for _, row in table_n])      # (refl, n_r, n_t)
    errs = np.sqrt(np.array([row[1] for _, row in table_n]))
    strong = (errs[:, ref, ref] > 0) & (nets[:, ref, ref] > 10.0*errs[:, ref, ref])
    say("   %-14s ellipsoid size, measured on %d strong reflections"
        % (tag, int(strong.sum())))
    if strong.sum() < 5:
        say("   %-14s too few - kept at %.1f x %.1f sigma"
            % ("", PEAK_SIGMAS_R, PEAK_SIGMAS_T))
        SHARED["peak_sigmas"] = (PEAK_SIGMAS_R, PEAK_SIGMAS_T)
        return SHARED["peak_sigmas"]
    q = np.array([SHARED.get("q_of", {}).get(index, 0.0)
                  for index, _ in table_n])
    n_t = _saturation(tag, "ACROSS Q", nets[:, -1, :], strong, q,
                      "along Q at %.1f sigma" % SATURATION_N[-1])
    k_t = SATURATION_N.index(n_t)
    n_r = _saturation(tag, "ALONG Q", nets[:, :, k_t], strong, q,
                      "across Q at %.1f sigma" % n_t)
    PEAK_SIGMAS_R, PEAK_SIGMAS_T = n_r, n_t
    SHARED["peak_sigmas"] = (n_r, n_t)
    say("   %-14s -> %.1f sigma along Q x %.1f sigma across, for every run"
        % ("", n_r, n_t))
    return SHARED["peak_sigmas"]


def _integrate_one(tag, state, lam, theta_lo, theta_hi):
    norm_all = np.linalg.norm(state["q"], axis=1)
    predicted = ws(tag, "predicted")
    # the d range is what the BANK and the BAND reach, not what the peak
    # search happened to find: taking it from the found peaks means integrating
    # only the reflections already seen, which is the opposite of the point
    two_theta = bank_theta(state)
    sin_hi = np.sin(np.radians(min(theta_hi, two_theta.max())/2.0))
    sin_lo = np.sin(np.radians(max(theta_lo, two_theta.min())/2.0))
    d_min = lam[0]/(2*max(sin_hi, 1e-6))
    d_max = lam[1]/(2*max(sin_lo, 1e-6))
    if D_RANGE:
        d_min, d_max = D_RANGE
    condition = CENTRING_NAMES.get((CENTRING or "P").upper(), "Primitive")
    tick("PredictPeaks for %s, d %.3f .. %.3f A, %s"
         % (tag, d_min, d_max, condition))
    # Predict from the MD workspace of THIS run, not from its peak list: the
    # peak list holds LeanElasticPeaks, which carry neither the instrument nor
    # the run's goniometer, so PredictPeaks on them used the identity and put
    # every reflection of a run at omega != 0 in the wrong place (runs 2 and 3
    # came back with centroids that did not move and I/sigma 0).  The MD
    # workspace carries both; it is given the run's UB first.
    source = state["md"]
    try:
        ub = mtd[state["peaks"]].sample().getOrientedLattice().getUB()
        call("SetUB", Workspace=source,
             UB=[float(v) for v in np.asarray(ub).ravel()])
    except Exception as trouble:
        say("   %-14s could not put UB on %s (%s); predicting from the"
            " peak list" % (tag, source, trouble))
        source = state["peaks"]
    call("PredictPeaks", InputWorkspace=source,
         OutputWorkspace=predicted, WavelengthMin=lam[0], WavelengthMax=lam[1],
         MinDSpacing=float(d_min), MaxDSpacing=float(d_max),
         ReflectionCondition=condition)
    table = mtd[predicted]
    if table.getNumberPeaks() == 0:
        say("   %-14s nothing predicted" % tag)
        return None
    q = np.array([list(p.getQSampleFrame()) for p in table])
    norm = np.linalg.norm(q, axis=1)
    # centroid the PREDICTED positions first: any residual error in UB puts the
    # box off the peak, and the distribution of these shifts is itself a direct
    # measure of how good UB is
    tick("CentroidPeaksMD on %d predicted reflections"
         % table.getNumberPeaks())
    radius = (float(CENTROID_RADIUS) if CENTROID_RADIUS else
              float(2.0*sigma_of(SHARED["sigma_radial"], float(norm.mean()))))

    def centroid():
        before = np.array([list(p.getQSampleFrame()) for p in mtd[predicted]])
        call("CentroidPeaksMD", InputWorkspace=state["md"],
             PeaksWorkspace=predicted, OutputWorkspace=predicted,
             PeakRadius=radius)
        after = np.array([list(p.getQSampleFrame()) for p in mtd[predicted]])
        if len(after) == len(before):
            moved = np.linalg.norm(after - before, axis=1)
            say("   %-14s centroid radius %.3f A-1: the centroids moved a"
                " median %.4f A-1, 90 %% below %.4f"
                % ("", radius, float(np.median(moved)),
                   float(np.quantile(moved, 0.9))))

    centroid()
    say("   %-14s that is the quality of UB, in the units the resolution"
        % "")
    say("   %-14s function is measured in" % "")
    if UB_PER_RUN:
        # sx_magic's stage_integrate: refine this run's UB on its centroided
        # predictions, then predict and centroid again so the boxes sit on
        # the refined positions (plan section 94)
        try:
            base = mtd[predicted].sample().getOrientedLattice()
            base = [base.a(), base.b(), base.c(), base.alpha(), base.beta(),
                    base.gamma()]
            call("FindUBUsingIndexedPeaks", PeaksWorkspace=predicted,
                 Tolerance=0.12, CommonUBForAll=False)
            own = mtd[predicted].sample().getOrientedLattice()
            say("   %-14s per-run UB: refined cell %.5f %.5f %.5f"
                % ("", own.a(), own.b(), own.c()))
            drift = max(abs(own.a()/base[0] - 1), abs(own.b()/base[1] - 1),
                        abs(own.c()/base[2] - 1))
            if drift > MAX_CELL_DRIFT:
                # sx_magic section 106: an axis the run hardly samples
                # (run 3, h = 0) drifts; keep the orientation only
                say("   %-14s an edge moved %.1f %% (> %.1f %%): orientation"
                    " only, on the common cell" % ("", 100*drift,
                                                   100*MAX_CELL_DRIFT))
                call("CalculateUMatrix", PeaksWorkspace=predicted,
                     a=base[0], b=base[1], c=base[2], alpha=base[3],
                     beta=base[4], gamma=base[5])
                own = mtd[predicted].sample().getOrientedLattice()
            call("SetUB", Workspace=source,
                 UB=[float(v) for v in np.asarray(own.getUB()).ravel()])
            call("PredictPeaks", InputWorkspace=source,
                 OutputWorkspace=predicted, WavelengthMin=lam[0],
                 WavelengthMax=lam[1], MinDSpacing=float(d_min),
                 MaxDSpacing=float(d_max), ReflectionCondition=condition)
            table = mtd[predicted]
            say("   %-14s predicted again: %d reflections"
                % ("", table.getNumberPeaks()))
            centroid()
        except Exception as trouble:
            say("   %-14s per-run UB failed (%s); integrating on the common"
                " UB" % ("", str(trouble).splitlines()[-1][:70]))
    result = ws(tag, "integrated")
    call("CloneWorkspace", InputWorkspace=predicted, OutputWorkspace=result)
    tick("integrating in the local frame, one ellipsoid per reflection")
    # Section 115: the box is binned ONCE per reflection in units of the one
    # resolution function (sigma_r along Q, sigma_t across), big enough for
    # the largest n tried, and the net intensity is taken for every n in
    # SATURATION_N from the same box: ellipsoid u^2 + rho^2 <= n^2, collar
    # |u| <= n, n < rho <= 1.8 n (LOCAL_COLLAR), in sigmas.
    n_box = max(SATURATION_N)
    half = (LOCAL_HALF[0]*n_box, LOCAL_HALF[1]*n_box)
    axes_ = [np.linspace(-h, h, m + 1)[:-1] + h/m
             for h, m in zip((half[0], half[1], half[1]), SATURATION_BINS)]
    u_, v1_, v2_ = np.meshgrid(*axes_, indexing="ij")
    rho_ = np.sqrt(v1_**2 + v2_**2)
    # section 117: every (n_r, n_t) pair, ellipsoid (u/n_r)^2 + (rho/n_t)^2 <= 1,
    # collar |u| <= n_r, n_t < rho <= 1.8 n_t; flat indices keep it fast
    masks = []
    for n_r in SATURATION_N:
        for n_t in SATURATION_N:
            ins = np.flatnonzero((u_/n_r)**2 + (rho_/n_t)**2 <= 1.0)
            col = np.flatnonzero((np.abs(u_) <= n_r)
                                 & (rho_ > LOCAL_COLLAR[0]*n_t)
                                 & (rho_ <= LOCAL_COLLAR[1]*n_t))
            masks.append((ins, col, float(len(ins))/max(float(len(col)), 1.0)))
    grid = (len(SATURATION_N), len(SATURATION_N))
    box = "%s_int" % PREFIX
    kept = 0
    table_n = []                    # per reflection: (net, var) for every n
    for index in range(mtd[result].getNumberPeaks()):
        peak = mtd[result].getPeak(index)
        vector = np.array(list(peak.getQSampleFrame()))
        modulus = float(np.linalg.norm(vector))
        if modulus <= 0.0:
            continue
        a = float(sigma_of(SHARED["sigma_radial"], modulus))
        b = float(sigma_of(SHARED["sigma_across"], modulus))
        radial, first, second = local_frame(vector)
        try:
            call("BinMD", InputWorkspace=state["md"], OutputWorkspace=box,
                 AxisAligned=False,
                 BasisVector0="r,A^-1,%.8f,%.8f,%.8f" % tuple(radial),
                 BasisVector1="t1,A^-1,%.8f,%.8f,%.8f" % tuple(first),
                 BasisVector2="t2,A^-1,%.8f,%.8f,%.8f" % tuple(second),
                 NormalizeBasisVectors=False,
                 Translation="%.8f,%.8f,%.8f" % tuple(vector),
                 OutputExtents=[-half[0]*a, half[0]*a, -half[1]*b, half[1]*b,
                                -half[1]*b, half[1]*b],
                 OutputBins=list(SATURATION_BINS))
            signal = np.array(mtd[box].getSignalArray(), dtype=float)
        finally:
            if box in mtd:
                DeleteWorkspace(box)
        signal[~np.isfinite(signal)] = 0.0
        flat = signal.ravel()
        net = np.empty(len(masks))
        var = np.empty(len(masks))
        for m, (ins, col, w) in enumerate(masks):
            gross = float(flat[ins].sum())
            background = float(flat[col].sum())
            net[m] = gross - w*background
            var[m] = max(gross + w*w*background, 0.0)
        table_n.append((index, (net.reshape(grid), var.reshape(grid))))
        SHARED.setdefault("q_of", {})[index] = modulus
        kept += 1
    if PEAK_SIGMAS_AUTO and "peak_sigmas" not in SHARED:
        choose_peak_sigmas(tag, table_n)
    nearest = lambda x: SATURATION_N.index(min(SATURATION_N,
                                               key=lambda n: abs(n - x)))
    k_r, k_t = nearest(PEAK_SIGMAS_R), nearest(PEAK_SIGMAS_T)
    for index, (net, var) in table_n:
        peak = mtd[result].getPeak(index)
        peak.setIntensity(float(net[k_r, k_t]))
        peak.setSigmaIntensity(float(ERROR_SCALE*np.sqrt(var[k_r, k_t])))
    state["table_n"] = dict(table_n)
    say("   %-14s %d reflections integrated; the ellipsoid is %.1f x %.1f"
        " sigma, so" % ("", kept, PEAK_SIGMAS_R, PEAK_SIGMAS_T))
    say("   %-14s %.4f x %.4f A-1 at |Q| = 1 and %.4f x %.4f at |Q| = 10"
        % ("", PEAK_SIGMAS_R*float(sigma_of(SHARED["sigma_radial"], 1.0)),
           PEAK_SIGMAS_T*float(sigma_of(SHARED["sigma_across"], 1.0)),
           PEAK_SIGMAS_R*float(sigma_of(SHARED["sigma_radial"], 10.0)),
           PEAK_SIGMAS_T*float(sigma_of(SHARED["sigma_across"], 10.0))))
    share = isolated_peaks(mtd[result])
    say("   %-14s %d of %d do not share their volume with a neighbour (%.0f %%)"
        % ("", int(share.sum()), len(share), 100*share.mean()))

    rows = []
    order = []
    for peak_index, p in enumerate(mtd[result]):
        vector = np.array(list(p.getQSampleFrame()))
        size = float(np.linalg.norm(vector))
        if size <= 0:
            continue
        sig_r, sig_t = sigmas(size)
        wavelength = float(p.getWavelength())
        angle = float(np.degrees(p.getScattering()))
        share = (_fraction_inside(wavelength, wavelength*sig_r/size, lam[0],
                                  lam[1])
                 * _fraction_inside(angle, float(np.degrees(2*sig_t/size)),
                                    theta_lo, theta_hi))
        rows.append((p.getH(), p.getK(), p.getL(), p.getIntensity(),
                     p.getSigmaIntensity(), 2*np.pi/size, wavelength, angle,
                     share, p.getDetectorID()))
        order.append(peak_index)
    if not rows:
        return None
    rows = np.array(rows, dtype=float)
    # section 112: a peak on the outermost segments/rows is partly off the bank
    edge = near_bank_edge(rows[:, 9])
    rows[edge, 8] = 0.0
    say("   %-14s %d reflections at the bank edges (within %d segments /"
        " %d rows) given coverage 0" % ("", int(edge.sum()), EDGE_SEGMENTS,
                                        EDGE_ROWS))
    raw = rows[:, 3].copy()
    if LORENTZ:
        # time-of-flight Laue: the Lorentz factor is lambda^4 / sin^2(theta),
        # and it varies by orders of magnitude across this band and this bank,
        # so without it the column is counts and reflections are not comparable
        theta = np.radians(rows[:, 7]/2.0)
        rows[:, 3] = rows[:, 3]*np.sin(theta)**2/rows[:, 6]**4
        rows[:, 4] = rows[:, 4]*np.sin(theta)**2/rows[:, 6]**4
    if SHARED.get("norm"):
        where = np.array([state["index_of"].get(int(v), 0) for v in rows[:, 9]])
        factor = normalisation_factor(where, rows[:, 6])
        safe = np.where(factor > 0, factor, 1.0)
        rows[:, 3] = rows[:, 3]/safe
        rows[:, 4] = rows[:, 4]/safe
        f_depth = normalisation_factor(where, rows[:, 6], "depth")
        f_groups = normalisation_factor(where, rows[:, 6], "groups")
    else:
        f_depth = f_groups = np.ones(len(rows))
    # columns 10-12, for checking offline (section 116): the counts before
    # Lorentz and normalisation, and what either normalisation would divide by
    rows = np.column_stack([rows, raw, f_depth, f_groups])
    # section 117: the whole (n_r, n_t) table, to try other ellipsoids offline;
    # intensity = net * scale for any cell of the grid
    try:
        scale = np.where(raw != 0, rows[:, 3]/np.where(raw != 0, raw, 1.0),
                         np.nan)
        tables = state.get("table_n", {})
        have = [i for i, k in enumerate(order) if k in tables]
        np.savez(out_name(tag, "npz").replace(".npz", "_shape.npz"),
                 n=np.array(SATURATION_N), rows=rows[have], scale=scale[have],
                 net=np.array([tables[order[i]][0] for i in have]),
                 var=np.array([tables[order[i]][1] for i in have]),
                 q=np.array([2*np.pi/rows[i, 5] for i in have]))
    except Exception as trouble:
        say("   shape table not written: %s" % trouble)
    state["rows"] = rows
    state["integrated"] = result
    keep = rows[:, 8] >= MIN_COVERAGE
    ratio = rows[:, 3]/np.maximum(rows[:, 4], 1e-9)
    say("   %-14s %4d predicted, %4d integrated, %4d with coverage >= %.1f,"
        % (tag, table.getNumberPeaks(), len(rows), int(keep.sum()),
           MIN_COVERAGE))
    say("   %-14s %4d of those with I/sigma > 3, median I/sigma %.1f"
        % ("", int((keep & (ratio > 3)).sum()),
           float(np.median(ratio[keep])) if keep.any() else 0.0))
    return result


def step_integrate():
    head("9. integrate every reflection the lattice predicts, in every run")
    if "cell" not in SHARED:
        need(step_cell, "there is no cell")
    if "cell" not in SHARED:
        return None
    lam = SHARED["lam"]
    theta_lo = SHARED["cut"] or min(bank_theta(state).min()
                                    for _, _, state in each_run())
    theta_hi = max(bank_theta(state).max() for _, _, state in each_run())
    sample = np.linspace(1.0, 10.0, 5)
    sig_r, sig_t = sigmas(sample)
    if PEAK_SIGMAS_AUTO and "peak_sigmas" not in SHARED:
        say("   the ellipsoid size is MEASURED on the first run (the n at which")
        say("   the strong peaks stop growing, along and across Q separately);")
        say("   shown here at %.1f x %.1f sigma, it runs from"
            % (PEAK_SIGMAS_R, PEAK_SIGMAS_T))
    else:
        say("   the ellipsoid is %.1f x %.1f sigma of the fitted function, so it"
            " runs from" % (PEAK_SIGMAS_R, PEAK_SIGMAS_T))
    say("   %.4f x %.4f A-1 at |Q| = 1 to %.4f x %.4f at |Q| = 10 - a"
        % (PEAK_SIGMAS_R*sig_r[0], PEAK_SIGMAS_T*sig_t[0],
           PEAK_SIGMAS_R*sig_r[-1], PEAK_SIGMAS_T*sig_t[-1]))
    say("   different shape for every reflection, in six bands of |Q|.")
    say("   Coverage is how much of that ellipsoid the measurement actually")
    say("   holds, given the band edges and the edges of the bank; a")
    say("   reflection at an edge has part of its peak outside the data and")
    say("   its integral is short by that much.")
    say("   Only one run is in Q space at a time here too: the MD workspace")
    say("   is rebuilt from the raw events, used, and dropped again.")
    say()
    for tag, _, state in each_run():
        if state.get("ub") is None:
            continue
        say("   %s" % tag)
        _prepare_one(tag, state, lam, SHARED["cut"], quiet=True)
        _build_md(tag, state, lam)
        try:
            _integrate_one(tag, state, lam, theta_lo, theta_hi)
        finally:
            _release(tag, state)


# ------------------------------------------------------------------ save --
def step_save():
    head("10. write the reflections out")
    if not any(state.get("rows") is not None for _, _, state in each_run()):
        if not need(step_integrate, "nothing has been integrated"):
            return None
    merged = []
    for tag, _, state in each_run():
        rows = state.get("rows")
        if rows is None:
            continue
        path = out_name(tag, "csv")
        with open(path, "w") as handle:
            handle.write("h,k,l,intensity,sigma,d,lambda,two_theta,coverage,"
                         "kept,detector,raw,norm_depth,norm_groups\n")
            for row in rows:
                handle.write("%d,%d,%d,%.4f,%.4f,%.5f,%.5f,%.3f,%.4f,%d,"
                             "%d,%.4f,%.6g,%.6g\n"
                             % (row[0], row[1], row[2], row[3], row[4], row[5],
                                row[6], row[7], row[8],
                                row[8] >= MIN_COVERAGE, row[9], row[10],
                                row[11], row[12]))
        say("   %-28s %4d reflections" % (path, len(rows)))
        merged.append((tag, rows))
        if state.get("integrated"):
            try:
                call("SaveReflections", InputWorkspace=state["integrated"],
                     Filename=os.path.abspath(out_name(tag, "int")),
                     Format="Fullprof")
                say("   %-28s Fullprof, for a refinement program"
                    % out_name(tag, "int"))
            except Exception as exc:
                say("   SaveReflections failed for %s: %s" % (tag, exc))
    if not merged:
        return None
    path = "%s_all.csv" % OUTPUT
    with open(path, "w") as handle:
        handle.write("run,h,k,l,intensity,sigma,d,lambda,two_theta,coverage,"
                     "kept,detector,raw,norm_depth,norm_groups\n")
        for tag, rows in merged:
            for row in rows:
                handle.write("%s,%d,%d,%d,%.4f,%.4f,%.5f,%.5f,%.3f,%.4f,%d,"
                             "%d,%.4f,%.6g,%.6g\n"
                             % (tag, row[0], row[1], row[2], row[3], row[4],
                                row[5], row[6], row[7], row[8],
                                row[8] >= MIN_COVERAGE, row[9], row[10],
                                row[11], row[12]))
    say("   %-28s every run, with a run column" % path)
    if len(merged) > 1:
        keys = {}
        for tag, rows in merged:
            for row in rows:
                if row[8] >= MIN_COVERAGE and row[4] > 0 and row[3] > 0:
                    keys.setdefault((row[0], row[1], row[2]),
                                    []).append((tag, row[3]))
        shared = {k: v for k, v in keys.items() if len(v) > 1}
        say()
        say("   %d reflections with a positive intensity in more than one run,"
            % len(shared))
        say("   out of %d rows written in all." % sum(len(r) for _, r in merged))
        if len(shared) < 10:
            say("   too few to say anything about agreement between runs.")
        else:
            scale = {tag: 1.0 for tag, _ in merged}
            for _ in range(20):
                for tag in scale:
                    ratios = [np.mean([v for other, v in group
                                       if other != tag])
                              / value
                              for group in shared.values()
                              for other, value in group if other == tag
                              and len(group) > 1]
                    if ratios:
                        scale[tag] *= float(np.median(ratios))**0.5
                middle = float(np.mean(list(scale.values())))
                for tag in scale:
                    scale[tag] /= middle
            say("   per-run scale factors from the common reflections: %s"
                % "  ".join("%s %.3f" % (tag, scale[tag]) for tag in scale))
            top, bottom = 0.0, 0.0
            for group in shared.values():
                values = [scale[tag]*value for tag, value in group]
                mean = float(np.mean(values))
                top += float(np.sum(np.abs(np.array(values) - mean)))
                bottom += float(np.sum(values))
            say("   R between runs on those, after scaling:  %.3f" % (top/max(bottom, 1e-9)))
            say("   This is a spread, not an agreement: 0 would be perfect.  It")
            say("   is the honest error bar on the whole chain, because nothing")
            say("   is shared between two runs except the instrument.  It is")
            say("   NOT R_int - that needs the Laue class of the space group to")
            say("   merge symmetry equivalents, which this workflow does not")
            say("   yet know.")
    if SHARED.get("cell"):
        cell = SHARED["cell"]
        say()
        say("   cell as measured: a=%.4f b=%.4f c=%.4f alpha=%.3f beta=%.3f "
            "gamma=%.3f" % tuple(cell))
        say("   Its SHAPE belongs to the crystal.  Its SCALE carries t0, which")
        say("   was %s." % ("%.0f us, from %s" % (T0, T0_SOURCE) if T0
                            else "zero - so the scale is uncalibrated"))
    return path


def step_tidy():
    head("11. what is left in memory")
    names = sorted(n for n in mtd.getObjectNames() if n.startswith(PREFIX))
    for name in names:
        say("   %-30s %8.0f MB" % (name, big(name)))
    dropped = 0
    if not KEEP_MD:
        for _, _, state in each_run():
            md = state.get("md")
            if md and md in mtd and big(md) > BIG_MB:
                size = big(md)
                DeleteWorkspace(md)
                dropped += 1
                say("   dropped %s (%.0f MB)" % (md, size))
    if dropped:
        say("   a workspace that size hangs the interface; --keep-md keeps them")
    else:
        say("   everything is kept, so every step can be looked at")


# ------------------------------------------------------------------ steps --
STEPS = [("load", step_load), ("propose", step_propose),
         ("prepare", step_prepare), ("peaks", step_peaks),
         ("resolution", step_resolution), ("index", step_index),
         ("cell", step_cell), ("normalise", step_normalise),
         ("integrate", step_integrate), ("save", step_save),
         ("tidy", step_tidy)]

USAGE = """magic_workflow.py FILE [FILE ...] [options]

Several NeXus files at once; one normalisation file for all of them.

  --cell A,B,C[,AL,BE,GA]  start from this cell; then the only unknown is
                       the orientation of each run.  Three numbers mean the
                       angles are 90; one number means a cube.  Whatever
                       equalities the cell carries (a=b, an angle of 90 or 120)
                       are KEPT through the refinement, so --system is not used
                       and not needed with it.
  --refine WHAT        none         take the cell and orientations as found
                       orientation  refine the orientations, hold the cell
                       cell         refine both              (default)
                       a,b          refine only these named cell parameters
                                    (and the orientations); c and the angles
                                    are held.  Names: a b c alpha beta gamma.
                                    Equalities in the given cell still hold, so
                                    naming a in a cell with a=b moves both.
  --centring LETTER    P A B C I F Robv Rrev - which reflections can exist.
                       Predicting and integrating with P on a centred lattice
                       spends half the work on reflections that cannot exist.
  --space-group NAME   e.g. Cmcm; only the centring letter is taken from it
  --d-range LO,HI      the d range to predict over; the default is what the
                       bank and the band reach, not what the peak search found
  --error-scale X      multiply every sigma by this.  McStas events are
                       WEIGHTED, so the counting error there is about 2.5 times
                       sqrt(N) and every I/sigma is otherwise overstated.
  --no-lorentz         leave the intensities as counts (not recommended: the
                       Lorentz factor lambda^4/sin^2(theta) varies by orders of
                       magnitude across the band and the bank)
  --system NAME        triclinic monoclinic orthorhombic tetragonal
                       rhombohedral hexagonal cubic   (default orthorhombic).
                       Only used when no --cell is given.
  --norm FILE          normalisation run, e.g. vanadium
  --norm-model M       depth (default, as sx_magic) or groups
  --idf FILE           instrument definition to use instead of the embedded one
  --gamma DEG          the angle bank A stood at in this run.  These files
                       carry it themselves, as entry/instrument/
                       detector_a_rotation, and it is read from there; this
                       option only overrides that.
  --gamma-b DEG        the same for bank B
  --idf-gamma-b DEG    what the geometry file carries for bank B.  Without it
                       bank B is left exactly as the file has it, because a
                       file that does not state its own angle cannot be
                       corrected.
  --idf-gamma DEG      the angle the geometry file already carries (default 0).
                       The rotation applied is +(gamma - idf-gamma), so if the
                       file was built from a run at this same angle, pass both
                       and nothing is turned.  A geometry file should carry the
                       bank at a reference angle and take the run's angle from
                       the run; this option exists because the file in use here
                       has the angle baked into absolute voxel positions.
  --bank-component N   the component --gamma turns; found in the file if absent
  --t0 US              t0 in microseconds, from a calibration.  Left out, the
                       instrument's 't0' parameter is used (3000 us in
                       MAGiC_Parameters.xml); 0 only if there is none
  --bank a|b           the bank whose events go into Q space   (default a).
                       A and B are different detectors and are never mixed;
                       the other one is masked.
  --lambda LO,HI       force the wavelength band instead of proposing one
  --two-theta-min DEG  force the angular mask
  --peak-sigmas X|R,T  size of the integration ellipsoid, in sigma (one
                               value for both axes, or along,across Q)
                               (default auto: where strong peaks saturate,
                               measured along and across Q separately)
  --centroid-radius R  A-1, centroiding of the predicted reflections
                               (default 0.15, as sx_magic; 0 = 2 sigma_r)
  --unwrap auto|MS     neutrons of the previous pulse: every event recorded
                       before the split goes to t + 71.43 ms, in the runs and
                       in the normalisation run alike.  'auto' finds the split
                       as the empty stretch inside the recorded frame (none =
                       nothing to unwrap); a number gives it in ms.  Needs
                       files converted with a --tof-window longer than the
                       frame edge (e.g. 22,92), see batch_convert.py
  --ub-per-run         refine each run's UB on its centroided predictions and
                       predict again before integrating, as sx_magic does
                       (default; a cell edge drifting > 3 %% keeps the
                       orientation only).  --common-ub switches it off
  --peak-distance X    FindPeaksMD merges peaks closer than this, A-1
                                                           (default 0.25)
  --min-coverage X     reject a reflection cut below this   (default 0.6)
  --index-peaks N      brightest peaks used to find the lattice  (default 45)
  --tolerance X        sigmas for calling an index integer  (default 2.5)
  --out STEM           stem for the files written           (default workflow)
  --steps a,b,c        run only these:
                       %s
  --ask                stop at every proposal instead of taking it
  --keep-md            keep the MD workspaces however large
  --max-warnings N     show at most N Warning lines from Mantid and numpy,
                       then suppress the rest with one notice   (default 20)
  --max-errors N       the same for Error lines                 (default 100)
                       -1 means no limit.  Every suppressed line still goes
                       to STEM_messages.log.  From a terminal this covers
                       Mantid's own log; in Workbench only the script output.
  --help

  a crystal simulated in three orientations (README_magic_workflow.md):
  python3 magic_workflow.py mysample_1.nxs mysample_2.nxs mysample_3.nxs \\
          --cell 2.89060,9.80240,12.58040 --centring C \\
          --t0 1715 --unwrap auto --lambda 0.65,2.25 --two-theta-min 20 \\
          --norm vanadium.nxs --idf MAGiC_Definition_vox.xml --out mysample
  a first look, stopping after the proposals:
  python3 magic_workflow.py mysample_1.nxs --t0 1715 --unwrap auto \\
          --steps load,propose
""" % " ".join(name for name, _ in STEPS)

EXTENSIONS = (".nxs", ".h5", ".hdf", ".hdf5", ".nx5", ".nexus")


def parse(argv):
    """Files, steps and options, in any order."""
    global RUNS, NORMALISATION, SYSTEM, IDF, BANK_GAMMA, T0, AUTO, KEEP_MD
    global T0_SOURCE, BANK, PEAK_DISTANCE
    global OUTPUT, TWO_THETA_MIN, LAMBDA_BAND, PEAK_SIGMAS, MIN_COVERAGE
    global PEAK_SIGMAS_R, PEAK_SIGMAS_T
    global PEAK_SIGMAS_AUTO, NORM_MODEL
    global INDEX_PEAKS, INDEX_TOLERANCE, CELL, REFINE, SYSTEM_GIVEN
    global REFINE_NAMES, CENTRING, D_RANGE, LORENTZ
    global CENTROID_RADIUS, UB_PER_RUN, UNWRAP
    global INDEX_SIGMA_T, ERROR_SCALE, BANK_COMPONENT, IDF_GAMMA
    global BANK_GAMMA_B, IDF_GAMMA_B, MAX_WARNINGS, MAX_ERRORS
    names = {name for name, _ in STEPS}
    files, steps = [], []
    i = 0
    while i < len(argv):
        item = argv[i]
        i += 1
        if item in ("--help", "-h"):
            say(USAGE)
            return None, None
        if item.startswith("--"):
            key = item[2:]
            value = None
            if "=" in key:
                key, value = key.split("=", 1)
            if key in ("ask", "keep-md", "no-lorentz", "ub-per-run",
                       "common-ub"):
                if key == "ask":
                    AUTO = False
                elif key == "keep-md":
                    KEEP_MD = True
                elif key == "ub-per-run":
                    UB_PER_RUN = True
                elif key == "common-ub":
                    UB_PER_RUN = False
                else:
                    LORENTZ = False
                continue
            if value is None:
                if i >= len(argv):
                    say("--%s needs a value" % key)
                    return None, None
                value = argv[i]
                i += 1
            if key == "centring":
                CENTRING = value.strip()
                if CENTRING.upper() not in CENTRING_NAMES:
                    say("--centring wants one of %s"
                        % " ".join(sorted(CENTRING_NAMES)))
                    return None, None
            elif key == "space-group":
                letter = value.strip()[:1].upper()
                CENTRING = {"R": "Robv"}.get(letter, letter)
                if CENTRING.upper() not in CENTRING_NAMES:
                    say("cannot read a centring from the space group %s" % value)
                    return None, None
                say("space group %s -> centring %s" % (value, CENTRING))
            elif key == "d-range":
                D_RANGE = tuple(sorted(float(v) for v in
                                       value.replace(",", " ").split()))
                if len(D_RANGE) != 2:
                    say("--d-range wants two numbers")
                    return None, None
            elif key == "cell":
                numbers = [float(v) for v in value.replace(",", " ").split()]
                if len(numbers) == 1:
                    numbers = numbers*3 + [90.0, 90.0, 90.0]
                elif len(numbers) == 3:
                    numbers = numbers + [90.0, 90.0, 90.0]
                if len(numbers) != 6:
                    say("--cell wants 1, 3 or 6 numbers")
                    return None, None
                CELL = numbers
            elif key == "refine":
                word = value.lower().strip()
                simple = {"none": "none", "no": "none",
                          "orientation": "orientation", "u": "orientation",
                          "cell": "cell", "all": "cell"}
                if word in simple:
                    REFINE, REFINE_NAMES = simple[word], None
                else:
                    names = [v for v in word.replace(",", " ").split() if v]
                    unknown = [n for n in names if n not in CELL_NAMES]
                    if unknown:
                        say("--refine does not know %s; names are %s, or one"
                            " of none, orientation, cell"
                            % (", ".join(unknown), " ".join(CELL_NAMES)))
                        return None, None
                    REFINE, REFINE_NAMES = "cell", set(names)
            elif key == "system":
                SYSTEM_GIVEN = True
                SYSTEM = value.lower()
                if SYSTEM not in FREE:
                    say("unknown system %s; one of %s"
                        % (value, " ".join(sorted(FREE))))
                    return None, None
            elif key == "norm":
                NORMALISATION = value
            elif key == "idf":
                IDF = value
            elif key == "gamma":
                BANK_GAMMA = float(value)
            elif key == "bank-component":
                BANK_COMPONENT = value
            elif key == "idf-gamma":
                IDF_GAMMA = float(value)
            elif key == "gamma-b":
                BANK_GAMMA_B = float(value)
            elif key == "idf-gamma-b":
                IDF_GAMMA_B = float(value)
            elif key == "t0":
                T0, T0_SOURCE = float(value), "the command line"
            elif key == "bank":
                BANK = value.strip().lower()
                if BANK not in ("a", "b"):
                    say("--bank wants a or b")
                    return None, None
            elif key == "lambda":
                LAMBDA_BAND = tuple(float(v) for v in value.split(","))
            elif key == "two-theta-min":
                TWO_THETA_MIN = float(value)
            elif key == "norm-model":
                if value.strip() not in ("depth", "groups"):
                    say("--norm-model wants depth or groups")
                    return None, None
                NORM_MODEL = value.strip()
            elif key == "peak-sigmas":
                if value.strip().lower() == "auto":
                    PEAK_SIGMAS_AUTO = True
                else:
                    parts = [float(v) for v in value.split(",")]
                    PEAK_SIGMAS_R = parts[0]
                    PEAK_SIGMAS_T = parts[-1]
                    PEAK_SIGMAS, PEAK_SIGMAS_AUTO = parts[0], False
            elif key == "unwrap":
                UNWRAP = value.strip()
                if UNWRAP.lower() != "auto":
                    try:
                        float(UNWRAP)
                    except ValueError:
                        say("--unwrap wants auto or a time in ms")
                        return None, None
            elif key == "centroid-radius":
                CENTROID_RADIUS = float(value)
            elif key == "peak-distance":
                PEAK_DISTANCE = float(value)
            elif key == "min-coverage":
                MIN_COVERAGE = float(value)
            elif key == "index-peaks":
                INDEX_PEAKS = int(value)
            elif key == "tolerance":
                INDEX_TOLERANCE = float(value)
                INDEX_SIGMA_T = float(value)
            elif key == "error-scale":
                ERROR_SCALE = float(value)
            elif key == "max-warnings":
                MAX_WARNINGS = int(value)
            elif key == "max-errors":
                MAX_ERRORS = int(value)
            elif key == "out":
                OUTPUT = value
            elif key == "steps":
                steps.extend(v for v in value.replace(",", " ").split() if v)
            else:
                say("unknown option --%s" % key)
                return None, None
            continue
        if item in names:
            steps.append(item)
        elif item.lower().endswith(EXTENSIONS) or os.path.exists(item):
            files.append(item)
        else:
            say("%s is neither a file that exists nor a step name" % item)
            say("steps: %s" % " ".join(sorted(names)))
            return None, None
    RUNS = files
    return files, steps


def main(argv=None):
    files, steps = parse(list(argv if argv is not None else sys.argv[1:]))
    if files is None:
        return 0
    CHATTER.start(MAX_WARNINGS, MAX_ERRORS, "%s_messages.log" % OUTPUT)
    try:
        return _run(files, steps)
    finally:
        CHATTER.stop()


def _run(files, steps):
    if not files:
        say(USAGE)
        say("no input file given")
        return 1
    missing = [f for f in files + ([NORMALISATION] if NORMALISATION else [])
               if not os.path.exists(f)]
    if missing:
        say("not here: %s" % ", ".join(missing))
        return 1
    table = dict(STEPS)
    wanted = steps or [name for name, _ in STEPS]
    unknown = [s for s in wanted if s not in table]
    if unknown:
        say("no such step: %s" % ", ".join(unknown))
        return 1
    say("files      : %s" % "  ".join(files))
    if CELL:
        say("cell       : a=%.4f b=%.4f c=%.4f alpha=%.3f beta=%.3f gamma=%.3f"
            % tuple(CELL))
        say("refine     : %s" % (",".join(sorted(REFINE_NAMES))
                                 if REFINE_NAMES else REFINE))
        if SYSTEM_GIVEN:
            say("system     : %s, IGNORED - a given cell already fixes it"
                % SYSTEM)
    else:
        say("cell       : not given, it will be searched for")
        say("system     : %s" % SYSTEM)
        say("refine     : %s" % REFINE)
    say("normalise  : %s" % (NORMALISATION or "none"))
    say("centring   : %s (%s)" % (CENTRING,
                                  CENTRING_NAMES.get(CENTRING.upper(), "?")))
    say("lorentz    : %s" % ("applied" if LORENTZ else "NOT applied"))
    say("bank       : %s only - the other bank is masked" % BANK.upper())
    if T0 is None:
        say("t0         : from the instrument's 't0' parameter, read at load")
    else:
        say("t0         : %.0f us%s" % (T0, "" if T0 else "  (uncalibrated)"))
    if T0 is not None and not T0:
        say("             with t0 = 0 every wavelength is off by t0*3.956e-3/L,")
        say("             so |Q| carries an error going as 1/lambda.  That is")
        say("             not a common scale - it distorts the cell shape.")
    say("steps      : %s" % " ".join(wanted))
    started = time.time()
    CHAIN[0] = len(wanted) > 1
    for name in wanted:
        table[name]()
    say()
    say("done in %.1f s" % (time.time() - started))
    return 0

# sys.exit(main(['fe4o5_1.nxs', 'fe4o5_2.nxs', 'fe4o5_3.nxs', '--steps','load', 'propose']))
# sys.exit(main(['fe4o5_1.nxs', 'fe4o5_2.nxs', 'fe4o5_3.nxs', '--system','orthorhombic', '--norm', 'vanadium.nxs', '--out', 'fe4o5']))
# NOTE: a bare sys.exit(main([...])) here runs at import time and IGNORES the
# command line, so every option typed in the shell is thrown away.  Keep such
# lines commented out and pass the arguments on the command line instead.
# sys.exit(main(['fe4o5_1.nxs', 'fe4o5_2.nxs', 'fe4o5_3.nxs', '--cell',
#                '2.89060,9.80240,12.58040', '--norm', 'vanadium.nxs',
#                '--centring', 'C', '--out', 'fe4o5']))

if __name__ == "__main__":
    sys.exit(main())
