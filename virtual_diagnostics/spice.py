"""Push a diagnostic signal through a real SPICE netlist.

Every electronic diagnostic in this package returns a ``(t, volts)`` pair, and
every one of them sits in front of signal conditioning that someone has already
designed in SPICE --- the amplifier, the filter, the cable, the delay line, the
limiter in front of the ADC.  :class:`SpiceFrontEnd` takes your netlist, drives
it with the beam-derived waveform, and hands back what comes out the other side.

This runs the **ngspice command-line binary** as a subprocess and reads its
rawfile.  There is no Python SPICE binding involved, which is deliberate:

- ngspice itself is actively released; the Python bindings are not.
- A subprocess is isolated.  A circuit that makes ngspice abort takes down a
  child process, not your Python session.
- Several runs go in parallel across cores with nothing shared between them.
- The only thing to install is a normal system package.

If ngspice is not installed, :func:`ngspice_available` returns ``False`` and
every call raises :class:`NgspiceNotFound` with installation instructions.
Nothing else in the package is affected --- the SPICE path is entirely optional.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

INSTALL_HINT = """\
ngspice was not found.

  Debian/Ubuntu   sudo apt install ngspice
  Fedora/RHEL     sudo dnf install ngspice
  Arch            sudo pacman -S ngspice
  macOS           brew install ngspice
  conda           conda install -c conda-forge ngspice
  Windows         https://ngspice.sourceforge.io/download.html
                  (add the directory holding ngspice_con.exe to PATH)

Then check it with:  ngspice -v

If it is installed somewhere unusual, point at it directly:
  export NGSPICE_EXECUTABLE=/path/to/ngspice\
"""

_CANDIDATE_PATHS = (
    "/usr/bin/ngspice",
    "/usr/local/bin/ngspice",
    "/opt/homebrew/bin/ngspice",
    "/opt/local/bin/ngspice",
)


class NgspiceNotFound(RuntimeError):
    """Raised when the ngspice binary cannot be located.

    Carries the installation instructions in its message, so a user who hits
    this in a script does not have to go looking for them.
    """

    def __init__(self, detail: str = "") -> None:
        super().__init__((detail + "\n\n" if detail else "") + INSTALL_HINT)


class NgspiceError(RuntimeError):
    """Raised when ngspice ran but did not produce usable results."""


def ngspice_executable() -> str | None:
    """Locate the ngspice binary.

    Honours ``NGSPICE_EXECUTABLE``, then ``PATH``, then a few usual places.

    Returns
    -------
    str or None
        Path to the executable, or ``None`` if it was not found.
    """
    configured = os.environ.get("NGSPICE_EXECUTABLE")
    if configured:
        return configured if os.path.exists(configured) else None
    for name in ("ngspice", "ngspice_con"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in _CANDIDATE_PATHS:
        if os.path.exists(candidate):
            return candidate
    return None


def ngspice_available() -> bool:
    """Whether a SPICE simulation can actually be run in this environment."""
    return ngspice_executable() is not None


def ngspice_version() -> str | None:
    """Version string reported by ``ngspice -v``, or ``None`` if unavailable."""
    executable = ngspice_executable()
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "-v"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in (result.stdout + result.stderr).splitlines():
        # ngspice decorates its banner with asterisks.
        cleaned = line.strip().strip("*").strip()
        if "ngspice" in cleaned.lower() and any(ch.isdigit() for ch in cleaned):
            return cleaned
    return None


def read_rawfile(path: str | os.PathLike) -> dict[str, NDArray]:
    """Parse an ngspice rawfile into ``{variable_name: values}``.

    Handles both the binary and ASCII layouts, and both real (transient, DC) and
    complex (AC) data.

    Parameters
    ----------
    path : path-like
        The rawfile written by ``ngspice -r``.

    Returns
    -------
    dict
        Variable name (lower-cased, as ngspice writes it --- ``"time"``,
        ``"v(out)"``, ``"i(vin)"``) mapped to a 1-D array.  Complex analyses give
        complex arrays.

    Raises
    ------
    NgspiceError
        If the file is missing, truncated, or has no recognisable header.

    Notes
    -----
    Only the **last** plot in the file is returned.  ngspice appends a plot per
    analysis, and the one you asked for is the one that ran last.
    """
    raw = Path(path).read_bytes()
    marker_binary = raw.rfind(b"Binary:\n")
    marker_ascii = raw.rfind(b"Values:\n")
    if marker_binary < 0 and marker_ascii < 0:
        raise NgspiceError(
            f"{path} is not an ngspice rawfile (no 'Binary:' or 'Values:' marker)."
        )
    is_binary = marker_binary > marker_ascii
    marker = marker_binary if is_binary else marker_ascii

    header_start = raw.rfind(b"Title:", 0, marker)
    header = raw[max(header_start, 0) : marker].decode("utf-8", errors="replace")

    variables: list[str] = []
    n_points = 0
    complex_data = False
    in_variables = False
    for line in header.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("flags:"):
            complex_data = "complex" in lowered
        elif lowered.startswith("no. points:"):
            n_points = int(stripped.split(":", 1)[1])
        elif lowered.startswith("variables:"):
            in_variables = True
        elif in_variables and stripped:
            fields = stripped.split()
            if len(fields) >= 2 and fields[0].isdigit():
                variables.append(fields[1].lower())

    if not variables or n_points <= 0:
        raise NgspiceError(f"{path} has no variables or no points in its header.")

    n_vars = len(variables)
    body = raw[marker + len(b"Binary:\n" if is_binary else b"Values:\n") :]

    if is_binary:
        per_value = 2 if complex_data else 1
        expected = n_points * n_vars * per_value
        # Trim to a whole number of doubles first: frombuffer raises on a short
        # or ragged buffer, which would hide the real problem.
        values = np.frombuffer(body[: (len(body) // 8) * 8], dtype="<f8")
        if values.size < expected:
            raise NgspiceError(
                f"{path} is truncated: expected {expected} doubles, got {values.size}."
            )
        values = values[:expected]
        if complex_data:
            values = values[0::2] + 1j * values[1::2]
        table = values.reshape(n_points, n_vars)
    else:
        numbers: list[complex] = []
        for line in body.decode("utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            token = fields[1] if fields[0].isdigit() and len(fields) > 1 else fields[0]
            numbers.append(complex(token.replace(",", "+") + "j") if complex_data else float(token))
        table = np.asarray(numbers).reshape(n_points, n_vars)

    return {name: np.ascontiguousarray(table[:, i]) for i, name in enumerate(variables)}


def _lookup(results: dict[str, NDArray], name: str) -> NDArray | None:
    """Find a node in rawfile results, tolerating the ``v(...)`` wrapper."""
    lowered = name.lower()
    for key in (lowered, f"v({lowered})", f"i({lowered})"):
        if key in results:
            return results[key]
    return None


def _format_pwl(t: NDArray[np.floating], v: NDArray[np.floating]) -> str:
    """Inline PWL data for a SPICE source line."""
    return " ".join(f"{time:.9e} {value:.9e}" for time, value in zip(t, v))


def _decimate(
    t: NDArray[np.floating], v: NDArray[np.floating], max_points: int
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Thin a waveform to at most ``max_points`` samples, keeping the endpoints."""
    if t.size <= max_points:
        return t, v
    index = np.unique(np.linspace(0, t.size - 1, max_points).round().astype(int))
    return t[index], v[index]


@dataclass
class SpiceFrontEnd:
    """Signal conditioning described by a SPICE netlist.

    Raw diagnostic devices in this package produce electrode-level signals and
    stop there.  Everything after that --- terminations, delay lines, hybrids,
    filters, amplifiers, limiters --- lives in your netlist, where you have
    already designed it.  This class drives that netlist with the diagnostic's
    signals and hands back the conditioned result.

    All sources are driven in a **single** simulation, so the netlist is free to
    combine them.  That is what makes a delay-line combiner work: feed in the
    four electrode waveforms, let the netlist route two of them through
    transmission lines and sum them, and read the combined trace out of one node.

    Parameters
    ----------
    netlist : str or Path
        Netlist text, or a path to a ``.cir`` file.  It must contain a
        placeholder voltage source for each entry in ``sources``, and each node
        named in ``outputs``.  Any ``.tran`` line is rewritten to match the
        input record.
    sources : str or sequence of str
        Names of the source lines to drive, in the order the input waveforms
        arrive.  Their nodes are kept; only their values are replaced.
    outputs : str or sequence of str
        Node or nodes to return.  A single string gives back a plain array; a
        sequence gives back a dict keyed by node name.  Names are
        case-insensitive, and the ``v(...)`` wrapper ngspice uses is optional.
    max_points : int
        Most PWL points to inline per source.  A long record is decimated
        uniformly, endpoints kept.
    step_time : float or None
        Transient step.  ``None`` uses the input sample interval.
    end_time : float or None
        Transient end time.  ``None`` uses the record length.
    max_step : float or None
        Maximum timestep ngspice may take, the fourth ``.tran`` argument.
        ``None`` lets ngspice choose, which is usually right; set it if a fast
        edge is being stepped over.
    timeout : float
        Seconds to wait for ngspice before giving up.

    Examples
    --------
    One signal in, one out::

        front_end = SpiceFrontEnd(netlist="frontend.cir", sources="Vin", outputs="out")
        t_out, v_out = front_end.run(t, volts)

    Four electrodes into a delay-line combiner, one trace out::

        combiner = SpiceFrontEnd(
            netlist="delay_line_combiner.cir",
            sources=("Vb1", "Vb2", "Vb3", "Vb4"),
            outputs="sum",
        )
        signals = bpm.measure(beam, rng=0)
        t_out, trace = combiner.run(signals.t, signals.volts)

    Notes
    -----
    A SPICE transient analysis starts at ``t = 0`` and silently discards
    anything before it.  Beam time axes are centred on the bunch and run
    negative, so the record is shifted to start at zero on the way in and
    shifted back on the way out --- the times you get back line up with the
    times you passed in.

    The record is *not* extended for you.  If the netlist delays a signal by
    more than the record length, that pulse simply never arrives; make the
    diagnostic's ``duration`` long enough to hold everything the netlist does,
    or set ``end_time`` explicitly.
    """

    netlist: str | Path
    sources: str | Sequence[str] = "Vin"
    outputs: str | Sequence[str] = "out"
    max_points: int = 4000
    step_time: float | None = None
    end_time: float | None = None
    max_step: float | None = None
    timeout: float = 300.0

    @property
    def source_names(self) -> tuple[str, ...]:
        """Source names as a tuple, whether one or many were given."""
        if isinstance(self.sources, str):
            return (self.sources,)
        return tuple(self.sources)

    @property
    def output_names(self) -> tuple[str, ...]:
        """Output node names as a tuple, whether one or many were given."""
        if isinstance(self.outputs, str):
            return (self.outputs,)
        return tuple(self.outputs)

    def netlist_text(self) -> str:
        """The netlist as text, read from disk if ``netlist`` is a path."""
        candidate = Path(str(self.netlist))
        if "\n" not in str(self.netlist) and candidate.exists():
            return candidate.read_text()
        return str(self.netlist)

    def _as_mapping(self, signals) -> dict[str, NDArray[np.floating]]:
        """Normalise the input into ``{source_name: waveform}``."""
        names = self.source_names
        if isinstance(signals, dict):
            unknown = set(signals) - set(names)
            if unknown:
                raise ValueError(
                    f"signals for unknown sources {sorted(unknown)}; this front end "
                    f"drives {list(names)}"
                )
            missing = set(names) - set(signals)
            if missing:
                raise ValueError(f"no signal given for sources {sorted(missing)}")
            return {name: np.asarray(signals[name], dtype=float) for name in names}

        array = np.asarray(signals, dtype=float)
        if array.ndim == 1:
            array = array[None, :]
        if array.shape[0] != len(names):
            raise ValueError(
                f"got {array.shape[0]} waveforms but this front end drives "
                f"{len(names)} sources {list(names)}"
            )
        return dict(zip(names, array))

    def build(self, t, signals) -> str:
        """Return the netlist with every source driven and ``.tran`` set.

        Decimation to ``max_points`` happens here, so what this returns is
        exactly what ngspice is given --- the first thing to look at when a
        simulation misbehaves.
        """
        t = np.asarray(t, dtype=float)
        driven = self._as_mapping(signals)
        step = self.step_time if self.step_time is not None else float(t[1] - t[0])
        end = self.end_time if self.end_time is not None else float(t[-1])

        thinned: dict[str, NDArray[np.floating]] = {}
        thinned_t = t
        for name, waveform in driven.items():
            thinned_t, thinned[name] = _decimate(t, waveform, self.max_points)

        wanted = {name.lower(): name for name in self.source_names}
        seen: set[str] = set()
        lines: list[str] = []
        for line in self.netlist_text().splitlines():
            stripped = line.strip()
            tokens = stripped.split()
            key = tokens[0].lower() if tokens else ""
            if key in wanted:
                if len(tokens) < 3:
                    raise ValueError(f"source line {stripped!r} does not name two nodes")
                name = wanted[key]
                seen.add(name)
                lines.append(
                    f"{tokens[0]} {tokens[1]} {tokens[2]} "
                    f"PWL({_format_pwl(thinned_t, thinned[name])})"
                )
            elif stripped.lower().startswith(".tran"):
                continue  # rewritten below
            elif stripped.lower().startswith(".end") and not stripped.lower().startswith(".ends"):
                continue  # re-added below
            else:
                lines.append(line)

        missing = [name for name in self.source_names if name not in seen]
        if missing:
            raise ValueError(
                f"no source named {missing} in the netlist. Add a placeholder line "
                f"such as '{missing[0]} in 0 PWL(0 0)' at the point where the "
                "diagnostic signal enters."
            )

        tran = f".tran {step:.9e} {end:.9e}"
        if self.max_step is not None:
            tran += f" 0 {self.max_step:.9e}"
        lines.append(tran)
        lines.append(".end")
        return "\n".join(lines) + "\n"

    def simulate(self, netlist: str) -> dict[str, NDArray]:
        """Run a complete netlist through ngspice and return its rawfile vectors.

        Exposed so you can drive ngspice with a netlist this class did not build
        --- an ``.ac`` sweep, say.

        Raises
        ------
        NgspiceNotFound
            If the binary is not installed.  The message carries the
            installation instructions.
        NgspiceError
            If ngspice ran but produced no usable output; the message carries
            ngspice's own log.
        """
        executable = ngspice_executable()
        if executable is None:
            raise NgspiceNotFound("Cannot run the SPICE front end.")

        with tempfile.TemporaryDirectory(prefix="virtual_diagnostics_") as directory:
            work = Path(directory)
            circuit = work / "circuit.cir"
            rawfile = work / "out.raw"
            circuit.write_text(netlist)
            try:
                completed = subprocess.run(
                    [executable, "-b", "-r", str(rawfile), str(circuit)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as error:
                raise NgspiceError(
                    f"ngspice did not finish within {self.timeout} s. Reduce "
                    "max_points, shorten the record, or raise timeout."
                ) from error

            log = (completed.stdout + completed.stderr).strip()
            if not rawfile.exists():
                raise NgspiceError(
                    f"ngspice produced no rawfile (exit code {completed.returncode}).\n"
                    f"--- ngspice output ---\n{log}"
                )
            try:
                return read_rawfile(rawfile)
            except NgspiceError as error:
                raise NgspiceError(
                    f"{error}\nexit code {completed.returncode}\n"
                    f"--- ngspice output ---\n{log}"
                ) from None

    def run(self, t: ArrayLike, signals, resample: bool = True):
        """Drive the netlist and read the output node or nodes.

        Parameters
        ----------
        t : array_like
            Sample times in seconds.  Must be uniformly spaced and increasing.
        signals : array_like or dict
            One waveform per source.  A 1-D array for a single source, a 2-D
            array shaped ``(n_sources, n_samples)`` matched to ``sources`` in
            order --- which is exactly the shape of
            :attr:`~virtual_diagnostics.bpm.ElectrodeSignals.volts` --- or a
            dict keyed by source name.
        resample : bool, optional
            Interpolate the result back onto ``t``.  ngspice takes adaptive time
            steps, so leaving this on is what lets the output compose with the
            rest of the package.  Turn it off to see the raw solver grid.

        Returns
        -------
        t_out : ndarray
        result : ndarray or dict of ndarray
            An array when ``outputs`` is a single node name, otherwise a dict
            keyed by node name.
        """
        t = np.asarray(t, dtype=float)
        if t.ndim != 1 or t.size < 2:
            raise ValueError("t must be a 1-D array of at least two samples")
        if np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly increasing")
        driven = self._as_mapping(signals)
        for name, waveform in driven.items():
            if waveform.shape != t.shape:
                raise ValueError(
                    f"waveform for {name!r} has shape {waveform.shape}, expected {t.shape}"
                )

        offset = float(t[0])
        shifted = t - offset
        results = self.simulate(self.build(shifted, driven))

        time = _lookup(results, "time")
        if time is None:
            raise NgspiceError(
                f"no time vector in the results; got {sorted(results)}. "
                "Did the netlist run a transient analysis?"
            )
        time = np.real(time)

        conditioned: dict[str, NDArray[np.floating]] = {}
        for wanted in self.output_names:
            found = _lookup(results, wanted)
            if found is None:
                raise NgspiceError(
                    f"node {wanted!r} is not in the results. Available: {sorted(results)}"
                )
            values = np.real(found)
            conditioned[wanted] = np.interp(shifted, time, values) if resample else values

        t_out = t if resample else time + offset
        if isinstance(self.outputs, str):
            return t_out, conditioned[self.outputs]
        return t_out, conditioned
