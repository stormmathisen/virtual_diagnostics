"""Beam position monitors, from the readout down to the individual electrodes.

Three models live here, in increasing order of what they cost and what they buy.

:class:`BPM` is the readout-level one: true centroid plus charge-dependent
noise.  Fast, and the right choice when you want a plausible position reading
and nothing else.

:class:`ButtonBPM` and :class:`StriplineBPM` generate the signal induced on each
electrode.  They share their geometry through :class:`ElectrodePickup` and
differ only in how an electrode responds:

- A **button** is a small capacitively coupled disc.  Its response is the
  ``R * C`` high pass of the electrode into its load, so a bunch produces a fast
  spike followed by an undershoot.
- A **stripline** is a length of transmission line.  The bunch induces a pulse
  as it enters and an inverted one as it leaves, so the response is the
  *difference of two delayed copies* of the beam current.  That makes it a comb
  filter, directional, and completely DC-free.

Modelling electrodes individually buys the effects a readout-level model cannot
have: nonlinearity as the beam approaches the pipe wall, electrode-to-electrode
gain mismatch, and a real waveform per electrode, which you push through your own signal
conditioning with :class:`~virtual_diagnostics.spice.SpiceFrontEnd`.

The combining and shaping that happens after the electrodes --- delay lines,
hybrids, filters, amplifiers --- is not modelled here.  It belongs in a SPICE
netlist, where you have already designed it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from .beam import C_LIGHT, Beam
from .electronic import CurrentMonitor
from .noise import default_rng

DIAGONAL_ELECTRODES = (np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4)
"""The usual four-electrode layout, at 45 degrees to the horizontal.

Electrodes sit off-axis so that synchrotron radiation and the beam's own field
sweep past them rather than straight into them.
"""


class BpmReading(NamedTuple):
    """A single BPM acquisition."""

    x: float
    """Horizontal position in metres, or ``nan`` below the charge threshold."""
    y: float
    """Vertical position in metres, or ``nan`` below the charge threshold."""
    charge: float
    """Charge seen by the pickup in coulombs."""


@dataclass
class BPM:
    """A beam position monitor, modelled at the readout.

    Parameters
    ----------
    resolution : float
        Single-shot RMS position noise in metres, **at** ``reference_charge``.
    reference_charge : float
        Charge in coulombs at which ``resolution`` was specified.
    gain : float or tuple of float
        Scale error, dimensionless.  ``(gx, gy)`` for per-plane errors.
    offset : tuple of float
        Electrical centre offset in metres, added to the reading.  This is the
        difference between the electrical and magnetic centres, and it is the
        error that survives every alignment campaign.
    charge_threshold : float
        Below this charge the BPM reports ``nan`` instead of a number, which is
        what the real thing does and what your feedback loop has to survive.

    Notes
    -----
    Resolution scales as ``reference_charge / charge``: the pickup signal is
    proportional to charge while the amplifier noise is not, so a BPM that is
    excellent at 250 pC can be useless at 1 pC.

    This model is linear by construction.  For the nonlinearity near the pipe
    wall, use :class:`ButtonBPM` or :class:`StriplineBPM`.
    """

    resolution: float = 10e-6
    reference_charge: float = 100e-12
    gain: float | tuple[float, float] = 1.0
    offset: tuple[float, float] = (0.0, 0.0)
    charge_threshold: float = 1e-12

    def __post_init__(self) -> None:
        if np.isscalar(self.gain):
            self.gain = (float(self.gain), float(self.gain))
        if self.reference_charge <= 0:
            raise ValueError("reference_charge must be positive")

    def noise_at(self, charge: float) -> float:
        """RMS position noise in metres at a given bunch charge."""
        magnitude = abs(charge)
        if magnitude <= 0:
            return float("inf")
        return self.resolution * self.reference_charge / magnitude

    def measure(self, beam: Beam, rng: np.random.Generator | int | None = None) -> BpmReading:
        """Read the beam position."""
        generator = default_rng(rng)
        charge = beam.total_charge
        if abs(charge) < self.charge_threshold:
            return BpmReading(float("nan"), float("nan"), charge)
        sigma = self.noise_at(charge)
        return BpmReading(
            x=self.gain[0] * beam.centroid_x + self.offset[0] + float(generator.normal(0.0, sigma)),
            y=self.gain[1] * beam.centroid_y + self.offset[1] + float(generator.normal(0.0, sigma)),
            charge=charge,
        )


# -- wall current geometry ------------------------------------------------


def wall_current_fraction(
    x: float,
    y: float,
    pipe_radius: float,
    electrode_angles,
    electrode_width: float,
    samples: int = 129,
) -> NDArray[np.floating]:
    r"""Fraction of the wall image current intercepted by each electrode.

    For a beam at radius $r$ and angle $\theta$ inside a circular pipe of radius
    $b$, the image current density on the wall is the Poisson kernel

    $$\frac{\mathrm{d}I}{\mathrm{d}\phi}
      = \frac{I_\mathrm{b}}{2\pi}\,
        \frac{b^2 - r^2}{b^2 + r^2 - 2br\cos(\phi-\theta)}$$

    which integrates to $I_\mathrm{b}$ over the full circumference.  Each
    electrode intercepts the integral of this over its own angular width.

    Parameters
    ----------
    x, y : float
        Beam centroid position in metres.
    pipe_radius : float
        Beam pipe radius $b$ in metres.
    electrode_angles : array_like
        Angular position of each electrode centre, radians.
    electrode_width : float
        Angular width of an electrode, radians.  For a button of radius $a$ on a
        pipe of radius $b$, this is about $2a/b$; for a stripline it is the
        coverage angle quoted on the drawing.
    samples : int, optional
        Integration points per electrode.

    Returns
    -------
    ndarray
        Intercepted fraction for each electrode.  Sums to the total angular
        coverage, never to more than one.

    Raises
    ------
    ValueError
        If the beam is outside the pipe.

    Notes
    -----
    The kernel is evaluated at the bunch *centroid*, not per particle.  For a
    beam with circular symmetry that is not an approximation: the Poisson kernel
    is harmonic inside the pipe, so its average over any circularly symmetric
    distribution equals its value at the centre.  A strongly elliptical beam
    close to the wall picks up a correction of order
    $(\sigma_x^2 - \sigma_y^2)/b^2$, which this ignores.
    """
    radius = float(np.hypot(x, y))
    if radius >= pipe_radius:
        raise ValueError(
            f"beam centroid at r={radius:.4g} m is outside the pipe "
            f"(radius {pipe_radius:.4g} m)"
        )
    theta = float(np.arctan2(y, x))
    angles = np.atleast_1d(np.asarray(electrode_angles, dtype=float))

    offsets = np.linspace(-electrode_width / 2, electrode_width / 2, samples)
    phi = angles[:, None] + offsets[None, :]
    kernel = (pipe_radius**2 - radius**2) / (
        pipe_radius**2 + radius**2 - 2 * pipe_radius * radius * np.cos(phi - theta)
    )
    return np.trapezoid(kernel, offsets, axis=1) / (2 * np.pi)


class ElectrodeSignals(NamedTuple):
    """Per-electrode waveforms from a pickup."""

    t: NDArray[np.floating]
    """Sample times in seconds."""
    volts: NDArray[np.floating]
    """Shape ``(n_electrodes, n_samples)``, one waveform per electrode."""
    angles: NDArray[np.floating]
    """Angular position of each electrode, radians."""
    fractions: NDArray[np.floating]
    """Wall current fraction intercepted by each electrode, before gain errors."""


# Kept for readability at call sites that only ever deal with buttons.
ButtonSignals = ElectrodeSignals


@dataclass
class ElectrodePickup:
    """Geometry and position readout shared by button and stripline pickups.

    This holds everything that does not depend on how an electrode responds:
    where the electrodes are, how much wall current each intercepts, and how to
    turn a set of amplitudes into a position.  :class:`ButtonBPM` and
    :class:`StriplineBPM` add the response.

    Parameters
    ----------
    pipe_radius : float
        Beam pipe radius in metres.
    electrode_angles : tuple of float
        Angular position of each electrode centre in radians.  Defaults to
        :data:`DIAGONAL_ELECTRODES`.
    electrode_width : float
        Angular width of an electrode in radians.
    gains : tuple of float or None
        Per-electrode gain errors, dimensionless.  ``None`` means all electrodes
        are perfect.  Real ones differ by a few percent, and that mismatch
        appears directly as a position offset.
    sensitivity : float
        Correction to the geometric position sensitivity, dimensionless.
        Calibrate it with :meth:`calibrate_sensitivity`.

    Notes
    -----
    Position comes from projecting the per-electrode amplitudes onto the
    electrode angles,

    $$x = b\\,S\\,\\frac{\\sum_i V_i \\cos\\phi_i}{\\sum_i V_i},$$

    which reduces to the familiar difference-over-sum for a symmetric layout and
    works for any arrangement.  It is linear only for small offsets; the Poisson
    kernel makes it read progressively low as the beam approaches the wall,
    which is the whole reason to model electrodes individually.
    """

    pipe_radius: float = 20e-3
    electrode_angles: tuple[float, ...] = DIAGONAL_ELECTRODES
    electrode_width: float = 0.5
    gains: tuple[float, ...] | None = None
    sensitivity: float = 1.0

    def __post_init__(self) -> None:
        if self.pipe_radius <= 0:
            raise ValueError(f"pipe_radius must be positive, got {self.pipe_radius}")
        if not 0 < self.electrode_width < 2 * np.pi:
            raise ValueError(
                f"electrode_width must be in (0, 2 pi), got {self.electrode_width}"
            )
        self.electrode_angles = tuple(float(a) for a in self.electrode_angles)
        if len(self.electrode_angles) < 2:
            raise ValueError("a BPM needs at least two electrodes")
        if self.gains is not None and len(self.gains) != len(self.electrode_angles):
            raise ValueError(
                f"gains has {len(self.gains)} entries but there are "
                f"{len(self.electrode_angles)} electrodes"
            )

    @property
    def n_electrodes(self) -> int:
        """Number of electrodes."""
        return len(self.electrode_angles)

    @property
    def electrode_gains(self) -> NDArray[np.floating]:
        """Per-electrode gains as an array, all ones when none were given."""
        if self.gains is None:
            return np.ones(self.n_electrodes)
        return np.asarray(self.gains, dtype=float)

    def intercept_fractions(self, x: float, y: float) -> NDArray[np.floating]:
        """Wall current fraction on each electrode for a beam at ``(x, y)``."""
        return wall_current_fraction(
            x, y, self.pipe_radius, self.electrode_angles, self.electrode_width
        )

    def amplitudes(
        self, signals: ElectrodeSignals, readout: Literal["peak", "integral"] = "peak"
    ) -> NDArray[np.floating]:
        """Reduce each electrode's waveform to a single amplitude.

        ``"peak"`` takes the maximum, as a peak-detecting front end does.
        ``"integral"`` takes the area of the positive lobe, which is quieter but
        needs a stable gate.  A stripline has no DC response at all, so its full
        integral is zero and only the positive lobe carries anything.
        """
        if readout == "peak":
            return signals.volts.max(axis=1)
        if readout == "integral":
            return np.trapezoid(np.clip(signals.volts, 0.0, None), signals.t, axis=1)
        raise ValueError(f"readout must be 'peak' or 'integral', got {readout!r}")

    def position_from_amplitudes(self, amplitudes, angles=None) -> tuple[float, float]:
        """Project electrode amplitudes onto ``(x, y)`` in metres."""
        amplitudes = np.asarray(amplitudes, dtype=float)
        angles = np.asarray(self.electrode_angles if angles is None else angles, dtype=float)
        total = amplitudes.sum()
        if total <= 0:
            return float("nan"), float("nan")
        scale = self.pipe_radius * self.sensitivity
        return (
            float(scale * (amplitudes * np.cos(angles)).sum() / total),
            float(scale * (amplitudes * np.sin(angles)).sum() / total),
        )

    def position(
        self, signals: ElectrodeSignals, readout: Literal["peak", "integral"] = "peak"
    ) -> tuple[float, float]:
        """Position in metres from a set of electrode waveforms."""
        return self.position_from_amplitudes(self.amplitudes(signals, readout), signals.angles)

    def calibrate_sensitivity(self, offsets=None, axis: Literal["x", "y"] = "x") -> float:
        """Fit the small-signal sensitivity correction from pure geometry.

        Sweeps a beam across the aperture using the intercept fractions alone
        --- no waveforms, no noise --- and fits the slope of true against
        reported position near the centre.  Assign the result to
        :attr:`sensitivity` so that :meth:`position` reads correctly.

        Returns
        -------
        float
            The correction factor.  It is close to one for a symmetric layout
            with narrow electrodes and drifts away as they get wider.
        """
        if offsets is None:
            offsets = np.linspace(-0.1, 0.1, 11) * self.pipe_radius
        offsets = np.asarray(offsets, dtype=float)
        angles = np.asarray(self.electrode_angles)
        trig = np.cos(angles) if axis == "x" else np.sin(angles)

        reported = []
        for offset in offsets:
            x, y = (offset, 0.0) if axis == "x" else (0.0, offset)
            fractions = self.intercept_fractions(x, y) * self.electrode_gains
            # Report with sensitivity 1 so the fitted slope is the correction.
            reported.append(self.pipe_radius * (fractions * trig).sum() / fractions.sum())
        return float(np.polyfit(reported, offsets, 1)[0])


def _default_button_electrode() -> CurrentMonitor:
    return CurrentMonitor(
        rise_time=30e-12,
        droop_time=250e-12,
        transimpedance=50.0,
        noise=1e-4,
        sample_rate=1e12,
        duration=4e-9,
        pretrigger=0.15,
    )


def _default_stripline_electrode() -> CurrentMonitor:
    return CurrentMonitor(
        rise_time=20e-12,
        droop_time=None,
        transimpedance=50.0,
        noise=1e-4,
        sample_rate=1e12,
        duration=6e-9,
        pretrigger=0.1,
    )


@dataclass
class ButtonBPM(ElectrodePickup):
    """A capacitively coupled button pickup, one signal per electrode.

    Parameters
    ----------
    electrode : CurrentMonitor
        Response of one electrode and its cable.  A button is capacitively
        coupled, so its ``droop_time`` is the electrode ``R * C`` --- with a
        50 ohm load and a few picofarads that is of order 100 ps, which is what
        makes a button pulse a fast spike followed by an undershoot.  Its
        ``transimpedance`` is the load resistance.

    See Also
    --------
    ElectrodePickup : the geometry and position readout this inherits.

    Examples
    --------
    >>> bpm = ButtonBPM(pipe_radius=20e-3)                    # doctest: +SKIP
    >>> bpm.sensitivity = bpm.calibrate_sensitivity()         # doctest: +SKIP
    >>> signals = bpm.measure(beam, rng=0)                    # doctest: +SKIP
    >>> signals.volts.shape                                   # doctest: +SKIP
    (4, 4000)

    Notes
    -----
    Set the electrode ``sample_rate`` so that the sample interval sits well
    below the rise time --- ten times below is comfortable.  A one-pole response
    sampled at interval ``dt`` peaks at ``Q / (tau + dt)`` rather than
    ``Q / tau``, so an under-sampled electrode reports an amplitude that is too
    small.  *Position* is unaffected either way, because the error is common to
    every electrode and cancels in the ratio; it is the absolute volts that
    suffer.
    """

    electrode: CurrentMonitor = field(default_factory=_default_button_electrode)

    def measure(
        self, beam: Beam, rng: np.random.Generator | int | None = None
    ) -> ElectrodeSignals:
        """Generate the waveform induced on every button.

        The bunch is binned once and then scaled per electrode, so the cost is
        one histogram plus one filter per electrode.
        """
        generator = default_rng(rng)
        t, current = self.electrode.sampled_current(beam)
        fractions = self.intercept_fractions(beam.centroid_x, beam.centroid_y)
        volts = np.stack(
            [
                self.electrode.apply(fraction * gain * current, rng=generator)
                for fraction, gain in zip(fractions, self.electrode_gains)
            ]
        )
        return ElectrodeSignals(
            t=t, volts=volts, angles=np.asarray(self.electrode_angles), fractions=fractions
        )


@dataclass
class StriplineBPM(ElectrodePickup):
    """A stripline pickup: a directional coupler made of transmission line.

    A stripline electrode of length $L$ is matched to the pipe at both ends.
    The bunch induces a pulse as it passes the upstream gap, half of which
    travels back out of the upstream port immediately.  When the bunch reaches
    the downstream gap it induces an equal and opposite pulse, and the half of
    *that* travelling upstream arrives at the upstream port a round trip later.
    The upstream port therefore sees

    $$V(t) = \\tfrac{1}{2} Z_0 f_i \\left[ I(t) - I(t - \\tau) \\right],
      \\qquad \\tau = \\frac{L}{\\beta c} + \\frac{L}{c},$$

    two pulses of opposite sign separated by $\\tau$ (which is $2L/c$ for a
    relativistic beam).  Three consequences follow, and all three are what make
    a stripline a stripline:

    - **No DC response.** The two pulses have equal area, so the waveform
      integrates to zero no matter what the bunch looks like.
    - **A comb response.** The transfer function is
      $|H(f)| = 2|\\sin(\\pi f \\tau)|$, peaking at the quarter-wave frequency
      $f_0 = 1/(2\\tau) = c/(4L)$ and nulling at every multiple of $1/\\tau$.
      Striplines are cut to put $f_0$ where the electronics wants it.
    - **Directivity.** A beam going the other way comes out of the *other*
      port. That is how a stripline in a storage ring tells the two beams apart.

    Parameters
    ----------
    length : float
        Electrode length $L$ in metres.
    impedance : float
        Line impedance $Z_0$ in ohms.
    directivity : float
        Isolation of the unused port in dB.  A real stripline manages 20 to
        30 dB; ``inf`` is a perfect coupler.
    electrode : CurrentMonitor
        Bandwidth and noise of the electrode and its cable.  Its
        ``transimpedance`` should be the line impedance, and its ``droop_time``
        should be ``None`` --- a stripline is not RC-coupled, its shaping comes
        entirely from the delay difference.

    See Also
    --------
    ElectrodePickup : the geometry and position readout this inherits.

    Examples
    --------
    >>> bpm = StriplineBPM(pipe_radius=20e-3, length=100e-3)  # doctest: +SKIP
    >>> bpm.quarter_wave_frequency / 1e9                      # doctest: +SKIP
    749.48
    >>> signals = bpm.measure(beam, rng=0)                    # doctest: +SKIP

    Notes
    -----
    The record has to be long enough to hold both pulses: keep the electrode's
    ``duration`` comfortably longer than :attr:`round_trip_time`, or the second
    pulse falls off the end of the record and the waveform looks like a button's.
    """

    length: float = 100e-3
    impedance: float = 50.0
    directivity: float = 26.0
    electrode: CurrentMonitor = field(default_factory=_default_stripline_electrode)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.impedance <= 0:
            raise ValueError(f"impedance must be positive, got {self.impedance}")

    def round_trip_time(self, beta: float = 1.0) -> float:
        """Separation of the two pulses in seconds, ``L/(beta c) + L/c``."""
        return self.length / (beta * C_LIGHT) + self.length / C_LIGHT

    @property
    def quarter_wave_frequency(self) -> float:
        """Frequency of peak response in Hz, ``c / (4 L)`` for a fast beam."""
        return 1.0 / (2.0 * self.round_trip_time())

    def transfer_magnitude(self, frequency, beta: float = 1.0):
        """``|H(f)| = 2 |sin(pi f tau)|``, the comb response of the electrode.

        Useful for choosing a length: put :attr:`quarter_wave_frequency` where
        your front end has gain, and keep the nulls away from it.
        """
        tau = self.round_trip_time(beta)
        return 2.0 * np.abs(np.sin(np.pi * np.asarray(frequency, dtype=float) * tau))

    def measure(
        self,
        beam: Beam,
        rng: np.random.Generator | int | None = None,
        port: Literal["upstream", "downstream"] = "upstream",
    ) -> ElectrodeSignals:
        """Generate the waveform at one port of every stripline.

        Parameters
        ----------
        beam : Beam
        rng : Generator or int, optional
        port : {"upstream", "downstream"}
            Which port to read.  For a beam travelling downstream the signal
            comes out of the upstream port; the downstream port sees only what
            leaks through, set by :attr:`directivity`.
        """
        generator = default_rng(rng)
        t, current = self.electrode.sampled_current(beam)
        tau = self.round_trip_time(beam.relativistic_beta)
        if tau > (t[-1] - t[0]):
            raise ValueError(
                f"the electrode record ({(t[-1] - t[0]) * 1e9:.3f} ns) is shorter than the "
                f"round trip time ({tau * 1e9:.3f} ns); increase electrode.duration"
            )

        # The bunch enters, then leaves: two pulses of opposite sign, tau apart.
        shaped = 0.5 * (current - np.interp(t - tau, t, current, left=0.0, right=0.0))
        if port == "downstream":
            shaped = shaped * 10.0 ** (-self.directivity / 20.0)
        elif port != "upstream":
            raise ValueError(f"port must be 'upstream' or 'downstream', got {port!r}")

        fractions = self.intercept_fractions(beam.centroid_x, beam.centroid_y)
        volts = np.stack(
            [
                self.electrode.apply(fraction * gain * shaped, rng=generator)
                for fraction, gain in zip(fractions, self.electrode_gains)
            ]
        )
        return ElectrodeSignals(
            t=t, volts=volts, angles=np.asarray(self.electrode_angles), fractions=fractions
        )
