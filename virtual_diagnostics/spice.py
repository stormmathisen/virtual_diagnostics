"""Push a diagnostic signal through a real SPICE netlist.

Every electronic diagnostic in this package returns a ``(t, volts)`` pair, and
every one of them sits in front of signal conditioning that someone has already
designed in SPICE --- the amplifier, the filter, the cable, the delay line, the
limiter in front of the ADC.  :class:`SpiceFrontEnd` takes your netlist, drives
it with the beam-derived waveform, and hands back what comes out the other side.

Requires the ``spice`` extra (`PySpice`) and a working ngspice shared library.
When either is missing, :func:`ngspice_available` returns ``False`` and nothing
else in the package is affected --- the SPICE path is entirely optional.

Notes
-----
Two practical wrinkles are handled here so you do not have to.

*The library name.*  PySpice looks for ``libngspice.so``, but distributions ship
``libngspice.so.0`` and only provide the unversioned symlink in the ``-dev``
package.  :func:`ngspice_library_path` finds the versioned file and points
PySpice at it through ``NGSPICE_LIBRARY_PATH``.

*A benign banner treated as an error.*  PySpice 1.5 treats any ngspice message
on stderr that does not begin with ``Warning:`` as a failure, and ngspice 42
prints ``Using SPARSE 1.3 as Direct Linear Solver`` there on every run.  The
simulation succeeds; only the wrapper thinks otherwise.  This module decides
success by whether the run actually produced data, and surfaces the real stderr
if it did not.
"""

from __future__ import annotations

import ctypes.util
import glob
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

_SEARCH_DIRECTORIES = (
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
    "/usr/local/lib",
    "/opt/homebrew/lib",
    "/usr/local/opt/ngspice/lib",
)

_shared = None


def ngspice_library_path() -> str | None:
    """Locate the ngspice shared library, versioned name included.

    Returns
    -------
    str or None
        Path to the library, or ``None`` if nothing was found.  An existing
        ``NGSPICE_LIBRARY_PATH`` always wins.
    """
    configured = os.environ.get("NGSPICE_LIBRARY_PATH")
    if configured:
        return configured
    found = ctypes.util.find_library("ngspice")
    if found and os.path.isabs(found):
        return found
    for directory in _SEARCH_DIRECTORIES:
        matches = sorted(glob.glob(os.path.join(directory, "libngspice.so*")))
        matches += sorted(glob.glob(os.path.join(directory, "libngspice*.dylib")))
        if matches:
            return matches[0]
    return None


def ngspice_available() -> bool:
    """Whether a SPICE simulation can actually be run in this environment."""
    try:
        _shared_instance()
    except Exception:
        return False
    return True


def _shared_instance():
    """Return the process-wide ngspice instance, creating it on first use."""
    global _shared
    if _shared is not None:
        return _shared

    path = ngspice_library_path()
    if path and not os.environ.get("NGSPICE_LIBRARY_PATH"):
        # Must be set before PySpice imports, hence the local import below.
        os.environ["NGSPICE_LIBRARY_PATH"] = path

    from PySpice.Spice.NgSpice.Shared import NgSpiceShared

    _shared = NgSpiceShared.new_instance()
    return _shared


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
        case-insensitive.
    max_points : int
        Most PWL points to inline per source.  A long record is decimated
        uniformly, endpoints kept.  Inlining tens of thousands of points makes
        ngspice crawl for no benefit.
    step_time : float or None
        Transient step.  ``None`` uses the input sample interval.
    end_time : float or None
        Transient end time.  ``None`` uses the record length.

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

        thinned = {}
        thinned_t = t
        for name, waveform in driven.items():
            thinned_t, thinned[name] = _decimate(t, waveform, self.max_points)

        wanted = {name.lower(): name for name in self.source_names}
        seen = set()
        output_lines = []
        for line in self.netlist_text().splitlines():
            stripped = line.strip()
            tokens = stripped.split()
            key = tokens[0].lower() if tokens else ""
            if key in wanted:
                if len(tokens) < 3:
                    raise ValueError(f"source line {stripped!r} does not name two nodes")
                name = wanted[key]
                seen.add(name)
                pwl = _format_pwl(thinned_t, thinned[name])
                output_lines.append(f"{tokens[0]} {tokens[1]} {tokens[2]} PWL({pwl})")
            elif stripped.lower().startswith(".tran"):
                continue  # rewritten below
            elif stripped.lower().startswith(".end") and not stripped.lower().startswith(".ends"):
                continue  # re-added below
            else:
                output_lines.append(line)

        missing = [name for name in self.source_names if name not in seen]
        if missing:
            raise ValueError(
                f"no source named {missing} in the netlist. Add a placeholder line "
                f"such as '{missing[0]} in 0 PWL(0 0)' at the point where the "
                "diagnostic signal enters."
            )

        output_lines.append(f".tran {step:.9e} {end:.9e}")
        output_lines.append(".end")
        return "\n".join(output_lines) + "\n"

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

        Raises
        ------
        RuntimeError
            If ngspice is unavailable, or if the run produced no data --- the
            simulator's stderr is included in the message.
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

        try:
            ngspice = _shared_instance()
        except Exception as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "ngspice is not available. Install the 'spice' extra (PySpice) and "
                "the ngspice shared library, e.g. 'apt install libngspice0'."
            ) from error

        from PySpice.Spice.NgSpice.Shared import NgSpiceCommandError

        ngspice.load_circuit(self.build(shifted, driven))
        try:
            ngspice.run()
        except NgSpiceCommandError:
            # PySpice flags benign ngspice banners on stderr as failures. Decide
            # from whether the run produced data instead.
            pass

        plot = ngspice.plot(None, ngspice.last_plot)
        names = {name.lower(): name for name in plot.keys()}
        if "time" not in names:
            raise RuntimeError(
                f"SPICE run produced no transient data.\nstderr:\n{ngspice.stderr}"
            )

        t_raw = np.asarray(plot[names["time"]].to_waveform(), dtype=float)
        results = {}
        for wanted in self.output_names:
            if wanted.lower() not in names:
                raise RuntimeError(
                    f"node {wanted!r} is not in the results. Available: {sorted(names)}"
                )
            raw = np.asarray(plot[names[wanted.lower()]].to_waveform(), dtype=float)
            results[wanted] = raw if not resample else np.interp(shifted, t_raw, raw)

        t_out = t if resample else t_raw + offset
        if isinstance(self.outputs, str):
            return t_out, results[self.outputs]
        return t_out, results
