"""Electronic diagnostics: current monitors and coherent radiation.

These all share a shape --- the bunch induces a signal in a pickup, the pickup
has a finite bandwidth, and an amplifier chain adds a noise floor.  The
differences between an ICT, an FCT, a wall current monitor and a Faraday cup are
differences of *parameters*, not of physics, so they are one class here.

Beam position monitors live in :mod:`virtual_diagnostics.bpm`, which reuses
:class:`CurrentMonitor` as the response of a single button electrode.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

from .beam import Beam
from .noise import default_rng, quantise, read

RISE_TIME_TO_TAU = 2.1972245773362196
"""10--90 % rise time of a single-pole low pass, in units of its time constant.

``ln(9)``.  Instrument datasheets quote rise time; the filter needs ``tau``.
"""


@dataclass
class LogAmpReadout:
    """Logarithmic amplifier readout, as used on the CLARA ICTs.

    The IOC reports charge through ``Q = QCal * 10**(V / UCal)``, fitted by a
    straight line of ``log10(Q)`` against the log-amp voltage.  Refit constants
    against a Faraday cup drop straight in here.

    Parameters
    ----------
    qcal : float
        Charge at zero volts, in coulombs.
    ucal : float
        Volts per decade of charge.

    Examples
    --------
    >>> amp = LogAmpReadout(qcal=1e-12, ucal=0.5)
    >>> v = amp.voltage_from_charge(250e-12)
    >>> round(amp.charge_from_voltage(v) * 1e12, 6)
    250.0
    """

    qcal: float
    ucal: float

    def charge_from_voltage(self, voltage):
        """Charge in coulombs from log-amp voltage."""
        return self.qcal * 10.0 ** (np.asarray(voltage, dtype=float) / self.ucal)

    def voltage_from_charge(self, charge):
        """Log-amp voltage from charge in coulombs.

        The inverse is undefined for zero or negative charge; those give
        ``-inf`` and ``nan`` respectively, which is what a real log amp does
        when the beam is off.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.ucal * np.log10(np.asarray(charge, dtype=float) / self.qcal)


@dataclass
class CurrentMonitor:
    """A bandwidth-limited beam current pickup with a digitiser.

    Covers integrating current transformers, fast current transformers, wall
    current monitors and Faraday cups.  The response is a single-pole low pass
    (the rise time) followed by a single-pole high pass (the droop), which is
    the standard two-parameter description on every transformer datasheet.

    An ICT and an FCT differ only in where you set those two numbers.  Give a
    transformer a rise time much longer than the bunch and it integrates: a
    bunch of charge ``Q`` comes out as ``(Q / tau) * exp(-t / tau)``, whose peak
    is proportional to charge and whose area is exactly ``Q``.  That is the ICT.
    Give it a rise time shorter than the bunch and the output follows the
    instantaneous current.  That is the FCT or the WCM.

    Parameters
    ----------
    rise_time : float
        10--90 % rise time in seconds.
    droop_time : float or None
        High-pass time constant in seconds --- the ``1/e`` decay of a step.
        ``None`` for a Faraday cup or any genuinely DC-coupled device.
    transimpedance : float
        Output volts per amp of beam current.
    noise : float
        RMS amplifier noise in volts.
    sample_rate : float
        Digitiser sample rate in Hz.
    duration : float
        Length of the acquired record in seconds.
    pretrigger : float
        Fraction of the record before the bunch arrives.  The rest holds the
        response, so leave room for several droop times if you want to see it.
    bits : int or None
        Digitiser resolution.  ``None`` returns the analogue volts undigitised.
    full_scale : float
        Volts at the top ADC code.
    readout : LogAmpReadout or None
        Optional log-amp charge readout, giving the scalar PV an operator sees
        alongside the waveform.

    Examples
    --------
    >>> ict = CurrentMonitor(rise_time=20e-9, droop_time=2e-6, transimpedance=1.25)
    >>> t, v = ict.measure(beam, rng=0)            # doctest: +SKIP
    >>> ict.integrated_charge(t, v) * 1e12         # doctest: +SKIP
    249.6

    Notes
    -----
    The waveform is built from :meth:`~virtual_diagnostics.beam.Beam.current_profile`,
    binned at the digitiser sample rate.  A bunch far shorter than one sample
    lands in a single bin, which is the correct impulse for a monitor whose rise
    time is much longer --- exactly the regime an ICT works in.
    """

    rise_time: float = 1e-9
    droop_time: float | None = 1e-6
    transimpedance: float = 1.25
    noise: float = 1e-4
    sample_rate: float = 5e9
    duration: float = 1e-6
    pretrigger: float = 0.1
    bits: int | None = None
    full_scale: float = 1.0
    readout: LogAmpReadout | None = None

    def __post_init__(self) -> None:
        if self.rise_time <= 0:
            raise ValueError(f"rise_time must be positive, got {self.rise_time}")
        if self.sample_rate <= 0 or self.duration <= 0:
            raise ValueError("sample_rate and duration must be positive")
        if not 0.0 <= self.pretrigger < 1.0:
            raise ValueError(f"pretrigger must be in [0, 1), got {self.pretrigger}")

    @property
    def dt(self) -> float:
        """Sample interval in seconds."""
        return 1.0 / self.sample_rate

    @property
    def bandwidth(self) -> float:
        """Upper 3 dB cut-off in Hz, from the rise time."""
        return RISE_TIME_TO_TAU / (2 * np.pi * self.rise_time)

    def response(self, current: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply the instrument's impulse response to a sampled beam current.

        Exposed separately so you can drive it with a current waveform from
        somewhere other than a :class:`~virtual_diagnostics.beam.Beam`.
        """
        dt = self.dt
        tau_lp = self.rise_time / RISE_TIME_TO_TAU
        alpha = dt / (tau_lp + dt)
        out = lfilter([alpha], [1.0, -(1.0 - alpha)], current)
        if self.droop_time is not None:
            if self.droop_time <= 0:
                raise ValueError(f"droop_time must be positive or None, got {self.droop_time}")
            k = self.droop_time / (self.droop_time + dt)
            out = lfilter([k, -k], [1.0, -k], out)
        return out

    def time_base(self, beam: Beam) -> NDArray[np.floating]:
        """Sample times of the acquired record, seconds."""
        n = max(int(round(self.duration * self.sample_rate)), 1)
        start = beam.centroid_t - self.pretrigger * self.duration
        return start + (np.arange(n) + 0.5) * self.dt

    def sampled_current(
        self, beam: Beam
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Beam current binned onto this monitor's time base.

        Split out from :meth:`measure` so that a pickup made of several
        electrodes --- a :class:`~virtual_diagnostics.bpm.ButtonBPM` --- can bin
        the bunch once and then scale it per electrode.

        Returns
        -------
        t : ndarray
            Sample times in seconds, on the beam's time axis.
        current : ndarray
            Beam current in amperes.
        """
        t = self.time_base(beam)
        dt = self.dt
        _, current = beam.current_profile(dt=dt, t_range=(t[0] - dt / 2, t[-1] + dt / 2))
        return t, current[: t.size]

    def apply(
        self, current: NDArray[np.floating], rng: np.random.Generator | int | None = None
    ) -> NDArray[np.floating]:
        """Turn a sampled beam current into the instrument's output.

        Impulse response, then transimpedance, then the noise floor, then the
        digitiser if one is configured.
        """
        volts = self.transimpedance * self.response(current)
        volts = volts + read(volts.shape, self.noise, default_rng(rng))
        if self.bits is not None:
            return quantise(volts, bits=self.bits, full_scale=self.full_scale)
        return volts

    def measure(
        self, beam: Beam, rng: np.random.Generator | int | None = None
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Acquire a waveform.

        Returns
        -------
        t : ndarray
            Sample times in seconds, on the beam's time axis.
        volts : ndarray
            Monitor output.  Integer ADC codes instead if ``bits`` is set.
        """
        t, current = self.sampled_current(beam)
        return t, self.apply(current, rng=rng)

    def integrated_charge(self, t: NDArray[np.floating], volts: NDArray[np.floating]) -> float:
        """Charge in coulombs from the area under the waveform.

        Exact for a monitor with no droop; droop makes this read low, which is
        the real error an integrating monitor has and the reason a droop
        correction exists on the hardware.
        """
        return float(np.trapezoid(np.asarray(volts, dtype=float), t) / self.transimpedance)

    def peak_charge(self, volts: NDArray[np.floating], calibration: float) -> float:
        """Charge from pulse height, given ``calibration`` in volts per coulomb.

        The peak-height method used on the CLARA Faraday cups.  Faster than
        integrating and immune to baseline drift, but it assumes the pulse shape
        never changes.
        """
        return float(np.max(volts) / calibration)

    def measure_charge(
        self, beam: Beam, rng: np.random.Generator | int | None = None
    ) -> float:
        """Scalar charge reading through the log-amp readout, coulombs.

        Requires ``readout``.  This is the PV an operator sees, round-tripped
        through the log amp, so it carries the log amp's calibration error
        rather than the waveform's noise.
        """
        if self.readout is None:
            raise ValueError("measure_charge needs a LogAmpReadout; set monitor.readout")
        t, volts = self.measure(beam, rng=rng)
        return float(
            self.readout.charge_from_voltage(
                self.readout.voltage_from_charge(self.integrated_charge(t, volts))
            )
        )


@dataclass
class CoherentRadiationMonitor:
    """A coherent radiation (CTR/CDR) bunch length monitor.

    Coherent emission scales as ``N**2 |F(f)|**2``, where ``F`` is the
    normalised Fourier transform of the longitudinal profile.  Short bunches
    radiate into higher frequencies, so a detector watching a fixed band gives a
    signal that rises steeply as the bunch compresses.  It is not a bunch length
    measurement on its own --- it is the compression tuning signal every linac
    optimiser actually uses.

    Parameters
    ----------
    band : tuple of float
        Detector passband ``(f_low, f_high)`` in Hz.
    calibration : float
        Volts per unit of ``Q**2 <|F|**2>``, with ``Q`` in coulombs.
    noise : float
        RMS detector noise in volts.
    oversampling : int
        Samples per period at ``f_high`` used when binning the profile.  Binning
        is a boxcar of width ``dt``, which rolls the response off as
        ``sinc(pi f dt)``; eight samples per period keeps that under a percent.

    Notes
    -----
    The single-particle emission spectrum is taken as flat across the band, and
    diffraction, the detector response shape and the transport optics are all
    folded into ``calibration``.  This gives the right *scaling* with bunch
    length, which is what a compression scan needs; it is not an absolute
    radiated power.
    """

    band: tuple[float, float] = (0.3e12, 3e12)
    calibration: float = 1.0
    noise: float = 0.0
    oversampling: int = 8

    def __post_init__(self) -> None:
        low, high = self.band
        if not 0 <= low < high:
            raise ValueError(f"band must be (low, high) with 0 <= low < high, got {self.band}")

    def form_factor(
        self, beam: Beam
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Longitudinal form factor.

        Returns
        -------
        frequency : ndarray
            Frequencies in Hz.
        magnitude_squared : ndarray
            ``|F(f)|**2``, normalised so ``|F(0)| == 1``.
        """
        dt = 1.0 / (self.oversampling * self.band[1])
        span = max(10.0 * beam.sigma_t, 20 * dt)
        centre = beam.centroid_t
        t, current = beam.current_profile(dt=dt, t_range=(centre - span, centre + span))
        spectrum = np.fft.rfft(current) * dt
        frequency = np.fft.rfftfreq(current.size, dt)
        charge = beam.total_charge
        if charge == 0:
            return frequency, np.zeros_like(frequency)
        return frequency, np.abs(spectrum / charge) ** 2

    def measure(self, beam: Beam, rng: np.random.Generator | int | None = None) -> float:
        """Detector output in volts."""
        frequency, magnitude_squared = self.form_factor(beam)
        low, high = self.band
        in_band = (frequency >= low) & (frequency <= high)
        if not np.any(in_band):
            raise ValueError(
                "No frequency samples fall in the detector band; the bunch is "
                "too long for this band, or oversampling is too low."
            )
        signal = self.calibration * beam.total_charge**2 * float(magnitude_squared[in_band].mean())
        return signal + float(read((), self.noise, default_rng(rng)))
