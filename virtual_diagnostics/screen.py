"""Scintillator / OTR screens and the two instruments built on top of them.

A screen turns the transverse charge density into a camera frame.  Everything
between the charge and the ADC code is lumped into knobs you can calibrate
against a real device, because that is how screens are characterised in the
control room: point size for the PSF, and total image counts against an ICT
reading for the sensitivity.

:class:`Spectrometer` and :class:`StreakedScreen` are thin wrappers.  They
contain **no new physics** --- the dipole and the transverse deflecting cavity
are your tracking code's job.  Track the beam to the screen, then use these to
turn pixel positions into energy or time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from .beam import Beam
from .noise import default_rng, quantise, read, shot


_ROI_ITERATIONS = 5
"""Shrink steps used to settle the region of interest onto the beam."""


def _weighted_moments(
    axis: NDArray[np.floating], weights: NDArray[np.floating]
) -> tuple[float, float]:
    """Centroid and RMS from signed weights, without any clipping."""
    total = weights.sum()
    if total <= 0:
        return float("nan"), float("nan")
    centroid = float(np.average(axis, weights=weights))
    variance = float(np.average((axis - centroid) ** 2, weights=weights))
    if variance <= 0:
        return centroid, float("nan")
    return centroid, float(np.sqrt(variance))


def profile_moments(
    axis: NDArray[np.floating],
    profile: NDArray[np.floating],
    background: float = 0.0,
    roi_sigma: float | None = 4.0,
) -> tuple[float, float]:
    """Background-subtracted centroid and RMS width of a 1-D profile.

    Parameters
    ----------
    axis : ndarray
        Coordinate of each sample.
    profile : ndarray
        Intensity at each sample.
    background : float, optional
        Level subtracted before weighting.  Residuals are kept **signed**.
    roi_sigma : float or None, optional
        Half-width of the region of interest, in units of the located beam
        width.  ``None`` uses the whole axis.

    Returns
    -------
    centroid, rms : float
        ``(nan, nan)`` if the profile carries no usable signal.

    Notes
    -----
    Two things go wrong when taking moments of a real frame, and this function
    exists to avoid both.

    *Rectifying the background-subtracted residual at zero* gives every empty
    bin a small positive weight.  Because the second moment weights bins by
    ``(x - mu)**2``, the tails then dominate and the RMS comes out badly
    inflated --- around 15 % on a typical frame here.

    *Keeping the residual signed but using the whole axis* is unbiased, but on a
    sensor much larger than the beam almost every bin is noise, and the variance
    of the estimate explodes.  It can even go negative and return a NaN.

    So: locate the beam with clipped (biased, but robust) estimates, iterated
    until the region of interest settles onto it, then take unbiased signed
    moments inside that region.  A single clipped pass is not enough --- on a
    sensor much larger than the beam it reads so wide that the region of
    interest covers the whole axis and restricts nothing.  Truncating a Gaussian
    at four sigma costs well under a percent, and the locator is biased wide,
    which keeps the region generous rather than tight.  This is what image
    analysis in a control room does by hand.
    """
    axis = np.asarray(axis, dtype=float)
    residual = np.asarray(profile, dtype=float) - background

    if roi_sigma is None:
        return _weighted_moments(axis, residual)

    # Locate the beam. A clipped estimate reads wide, and on a sensor much
    # larger than the beam it reads *very* wide, so shrink onto the beam by
    # re-estimating inside the region of interest until it settles.
    centre, width = _weighted_moments(axis, np.clip(residual, 0.0, None))
    for _ in range(_ROI_ITERATIONS):
        if not np.isfinite(width) or width <= 0:
            return _weighted_moments(axis, residual)
        inside = np.abs(axis - centre) <= roi_sigma * width
        if inside.sum() < 3:
            break
        centre, width = _weighted_moments(axis[inside], np.clip(residual[inside], 0.0, None))

    if not np.isfinite(width) or width <= 0:
        return _weighted_moments(axis, residual)
    inside = np.abs(axis - centre) <= roi_sigma * width
    if inside.sum() < 3:
        return centre, width
    return _weighted_moments(axis[inside], residual[inside])


def deconvolve(measured: float, *contributions: float) -> float:
    """Remove known widths from a measured RMS in quadrature.

    Returns ``0.0`` rather than a NaN when the contributions exceed the
    measurement, which happens routinely when the beam is smaller than the
    resolution.  A zero here means *unresolved*, not *zero width*.
    """
    variance = measured**2 - sum(c**2 for c in contributions)
    return float(np.sqrt(variance)) if variance > 0 else 0.0


@dataclass
class ScreenImage:
    """A camera frame plus the calibration needed to interpret it.

    Attributes
    ----------
    counts : ndarray
        Integer ADC codes, shape ``(ny, nx)``.  Row 0 is the *lowest* ``y``, so
        display it with ``origin="lower"``.
    x_axis, y_axis : ndarray
        Pixel centre coordinates in metres, in **beam** coordinates (any screen
        tilt has already been divided out), so widths measured from the image
        compare directly against :attr:`~virtual_diagnostics.beam.Beam.sigma_x`.
    background : float
        Dark level in counts, subtracted by default in the moment methods.
    off_screen_fraction : float
        Fraction of the bunch charge that missed the sensor.  Non-zero means
        your measured widths are truncated; check it before trusting them.
    saturated_fraction : float
        Fraction of pixels sitting on the top ADC code.
    pixel_size : tuple of float
        ``(dx, dy)`` in metres, in beam coordinates.
    psf_sigma : tuple of float
        ``(sigma_x, sigma_y)`` of the applied point spread function, metres, in
        beam coordinates.  Feed these to :func:`deconvolve`.
    """

    counts: NDArray[np.integer]
    x_axis: NDArray[np.floating]
    y_axis: NDArray[np.floating]
    background: float = 0.0
    off_screen_fraction: float = 0.0
    saturated_fraction: float = 0.0
    pixel_size: tuple[float, float] = (0.0, 0.0)
    psf_sigma: tuple[float, float] = (0.0, 0.0)

    def projection_x(self) -> NDArray[np.floating]:
        """Sum down the columns: intensity against ``x``."""
        return self.counts.sum(axis=0, dtype=float)

    def projection_y(self) -> NDArray[np.floating]:
        """Sum across the rows: intensity against ``y``."""
        return self.counts.sum(axis=1, dtype=float)

    def moments_x(
        self, background: float | None = None, roi_sigma: float | None = 4.0
    ) -> tuple[float, float]:
        """Centroid and RMS width in ``x``, metres.  See :func:`profile_moments`."""
        bg = self.background * self.counts.shape[0] if background is None else background
        return profile_moments(self.x_axis, self.projection_x(), bg, roi_sigma)

    def moments_y(
        self, background: float | None = None, roi_sigma: float | None = 4.0
    ) -> tuple[float, float]:
        """Centroid and RMS width in ``y``, metres.  See :func:`profile_moments`."""
        bg = self.background * self.counts.shape[1] if background is None else background
        return profile_moments(self.y_axis, self.projection_y(), bg, roi_sigma)

    def beam_size(
        self, deconvolved: bool = True, roi_sigma: float | None = 4.0
    ) -> tuple[float, float]:
        """Measured ``(sigma_x, sigma_y)`` in metres.

        With ``deconvolved=True`` the PSF and the pixel binning variance
        (``pitch**2 / 12``) are removed in quadrature, which is the closest this
        image gets to the true beam size.
        """
        _, rms_x = self.moments_x(roi_sigma=roi_sigma)
        _, rms_y = self.moments_y(roi_sigma=roi_sigma)
        if not deconvolved:
            return rms_x, rms_y
        return (
            deconvolve(rms_x, self.psf_sigma[0], self.pixel_size[0] / np.sqrt(12)),
            deconvolve(rms_y, self.psf_sigma[1], self.pixel_size[1] / np.sqrt(12)),
        )


@dataclass
class Screen:
    """A scintillator or OTR screen viewed by a digital camera.

    Parameters
    ----------
    pixel_size : float or tuple of float
        Pixel pitch projected onto the *screen surface*, metres.  A scalar
        applies to both axes.
    resolution : tuple of int
        ``(nx, ny)`` sensor size in pixels.
    centre : tuple of float
        ``(x, y)`` beam coordinate that lands on the middle of the sensor,
        metres.  This is the screen alignment offset.
    tilt : float
        Screen rotation about the vertical axis, radians.  A screen at 45
        degrees is ``np.pi / 4``.  Its only effect is that a pixel subtends
        ``pixel_size * cos(tilt)`` in beam ``x``.  If your camera views along
        the beam axis the factor already cancels --- leave this at zero and put
        everything in ``pixel_size``.
    psf_sigma : float
        Gaussian point spread of scintillator thickness plus optics, metres on
        the screen surface.  This is the dominant resolution limit on a thick
        YAG screen and the reason small beams cannot be measured on one.
    counts_per_pc : float
        Total ADC counts summed over the whole image per picocoulomb of charge
        on the screen.  One lumped number for light yield, collection solid
        angle, quantum efficiency and camera gain, because that product is what
        you calibrate against an ICT.
    gain : float
        ADC counts per photoelectron.  Sets the *shot noise* scale: the noise on
        a pixel is ``sqrt(gain * counts)``.  Raising ``counts_per_pc`` alone
        makes a brighter, quieter image; raising ``gain`` makes it noisier.
    dark_offset : float
        Pedestal in counts.
    read_noise : float
        RMS electronics noise in counts.
    bits : int
        ADC resolution.
    full_scale : float, optional
        Counts mapping to the top code.  Defaults to ``2**bits - 1``.

    Examples
    --------
    >>> screen = Screen(pixel_size=12e-6, resolution=(400, 300), psf_sigma=30e-6)
    >>> image = screen.measure(beam, rng=0)          # doctest: +SKIP
    >>> image.beam_size()                            # doctest: +SKIP
    (0.000101, 0.000198)

    Notes
    -----
    The pipeline is: bin charge onto pixels, blur with the PSF, convert to
    counts, add photon shot noise, add pedestal and read noise, then clip and
    quantise.  Blurring happens *before* the noise because the PSF is optical
    and the noise is detection --- doing it the other way round smooths the
    noise and gives a falsely clean image.
    """

    pixel_size: float | tuple[float, float] = 10e-6
    resolution: tuple[int, int] = (640, 480)
    centre: tuple[float, float] = (0.0, 0.0)
    tilt: float = 0.0
    psf_sigma: float = 0.0
    counts_per_pc: float = 1e5
    gain: float = 1.0
    dark_offset: float = 100.0
    read_noise: float = 5.0
    bits: int = 12
    full_scale: float | None = None

    def __post_init__(self) -> None:
        if np.isscalar(self.pixel_size):
            self.pixel_size = (float(self.pixel_size), float(self.pixel_size))
        if min(self.pixel_size) <= 0:
            raise ValueError(f"pixel_size must be positive, got {self.pixel_size}")
        if min(self.resolution) < 1:
            raise ValueError(f"resolution must be at least 1x1, got {self.resolution}")

    # -- geometry ---------------------------------------------------------

    @property
    def beam_pixel_size(self) -> tuple[float, float]:
        """Pixel pitch expressed in beam coordinates, metres."""
        return (self.pixel_size[0] * np.cos(self.tilt), self.pixel_size[1])

    @property
    def beam_psf_sigma(self) -> tuple[float, float]:
        """PSF sigma expressed in beam coordinates, metres."""
        return (self.psf_sigma * np.cos(self.tilt), self.psf_sigma)

    @property
    def axes(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Pixel centre coordinates ``(x_axis, y_axis)`` in beam metres."""
        nx, ny = self.resolution
        dx, dy = self.beam_pixel_size
        x = (np.arange(nx) - (nx - 1) / 2) * dx + self.centre[0]
        y = (np.arange(ny) - (ny - 1) / 2) * dy + self.centre[1]
        return x, y

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` of the sensor in beam metres."""
        nx, ny = self.resolution
        dx, dy = self.beam_pixel_size
        return (
            self.centre[0] - nx * dx / 2,
            self.centre[0] + nx * dx / 2,
            self.centre[1] - ny * dy / 2,
            self.centre[1] + ny * dy / 2,
        )

    # -- measurement ------------------------------------------------------

    def measure(self, beam: Beam, rng: np.random.Generator | int | None = None) -> ScreenImage:
        """Image a beam.

        Parameters
        ----------
        beam : Beam
            The distribution arriving at the screen.
        rng : Generator or int, optional
            Random source.  Pass an int for a reproducible shot, ``None`` for a
            fresh one.

        Returns
        -------
        ScreenImage
        """
        generator = default_rng(rng)
        nx, ny = self.resolution
        x_min, x_max, y_min, y_max = self.extent

        charge, _, _ = np.histogram2d(
            beam.x,
            beam.y,
            bins=(nx, ny),
            range=((x_min, x_max), (y_min, y_max)),
            weights=beam.q,
        )
        charge = charge.T  # (nx, ny) -> image (ny, nx)

        total = float(np.abs(beam.q).sum())
        on_screen = float(np.abs(charge).sum())
        off_screen_fraction = 1.0 - on_screen / total if total > 0 else 0.0

        counts = np.abs(charge) * 1e12 * self.counts_per_pc

        dx, dy = self.beam_pixel_size
        sigma_x, sigma_y = self.beam_psf_sigma
        if sigma_x > 0 or sigma_y > 0:
            # gaussian_filter axis order follows the array: (rows=y, cols=x).
            counts = ndimage.gaussian_filter(
                counts, sigma=(sigma_y / dy, sigma_x / dx), mode="constant"
            )

        # Shot noise lives on photoelectrons; counts = gain * photoelectrons.
        if self.gain > 0:
            counts = self.gain * shot(counts / self.gain, generator)
        counts = counts + self.dark_offset + read(counts.shape, self.read_noise, generator)

        codes = quantise(counts, bits=self.bits, full_scale=self.full_scale)
        top = 2**self.bits - 1
        x_axis, y_axis = self.axes

        return ScreenImage(
            counts=codes,
            x_axis=x_axis,
            y_axis=y_axis,
            background=self.dark_offset,
            off_screen_fraction=off_screen_fraction,
            saturated_fraction=float((codes >= top).mean()),
            pixel_size=(dx, dy),
            psf_sigma=(sigma_x, sigma_y),
        )


@dataclass
class Spectrometer:
    """A dispersive screen read out as an energy spectrum.

    Track the beam through your dipole in the tracking code, put this on the
    screen after it, and tell it the dispersion.  That is the whole instrument.

    Parameters
    ----------
    screen : Screen
        The screen in the dispersive arm.
    dispersion : float
        ``eta`` in metres: a particle with relative momentum deviation
        ``delta`` lands at ``x = eta * delta``.
    reference_energy : float
        Total energy in eV of a particle landing at ``x = 0``.
    axis : {"x", "y"}
        Which image axis is dispersive.

    Notes
    -----
    The measured width contains the betatron beam size at the screen as well as
    the energy spread.  Pass that size to :meth:`energy_spread` (get it from a
    zero-dispersion measurement, or from your optics) or the number you get back
    is an upper limit, not the energy spread.
    """

    screen: Screen
    dispersion: float
    reference_energy: float
    axis: Literal["x", "y"] = "x"

    def __post_init__(self) -> None:
        if self.dispersion == 0:
            raise ValueError("dispersion must be non-zero; a screen with no dispersion is a Screen")

    def measure(self, beam: Beam, rng: np.random.Generator | int | None = None) -> ScreenImage:
        """Image the beam.  Identical to ``self.screen.measure(beam, rng)``."""
        return self.screen.measure(beam, rng=rng)

    def _projection(self, image: ScreenImage):
        if self.axis == "x":
            return image.x_axis, image.projection_x(), image.counts.shape[0], 0
        return image.y_axis, image.projection_y(), image.counts.shape[1], 1

    def spectrum(self, image: ScreenImage) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Convert the dispersive projection into energy against intensity.

        Returns
        -------
        energy : ndarray
            Total energy in eV for each bin, ``E_ref * (1 + x / eta)``.
        intensity : ndarray
            Background-subtracted counts.  Not normalised.
        """
        position, profile, rows, _ = self._projection(image)
        energy = self.reference_energy * (1.0 + position / self.dispersion)
        intensity = np.clip(profile - image.background * rows, 0.0, None)
        order = np.argsort(energy)
        return energy[order], intensity[order]

    def energy_spread(
        self, image: ScreenImage, beam_size: float = 0.0, roi_sigma: float | None = 4.0
    ) -> float:
        """RMS energy spread in eV.

        Parameters
        ----------
        image : ScreenImage
        beam_size : float, optional
            RMS betatron size at the screen in metres, removed in quadrature
            along with the PSF and pixel pitch.  Leave at zero to get the upper
            limit.
        """
        position, profile, rows, index = self._projection(image)
        _, rms = profile_moments(position, profile, image.background * rows, roi_sigma)
        intrinsic = deconvolve(
            rms,
            beam_size,
            image.psf_sigma[index],
            image.pixel_size[index] / np.sqrt(12),
        )
        return self.reference_energy * intrinsic / abs(self.dispersion)

    def mean_energy(self, image: ScreenImage) -> float:
        """Centroid energy in eV."""
        position, profile, rows, _ = self._projection(image)
        centroid, _ = profile_moments(position, profile, image.background * rows)
        return self.reference_energy * (1.0 + centroid / self.dispersion)


@dataclass
class StreakedScreen:
    """A screen downstream of a transverse deflecting structure.

    The deflector maps arrival time onto transverse position with a shear
    ``S`` in metres per second, so the image is a picture of the longitudinal
    profile.  As with :class:`Spectrometer`, the deflecting cavity itself is
    your tracking code's job; this only calibrates the resulting image.

    Parameters
    ----------
    screen : Screen
        The screen downstream of the deflector.
    shear : float
        ``S`` in m/s: a particle arriving at time ``t`` lands at ``y = S * t``.
        For a cavity of voltage ``V`` at RF wavenumber ``k`` with ``sqrt(beta
        beta_s) sin(dphi)`` optics, ``S = c * k * V * sqrt(beta beta_s) *
        sin(dphi) / E``.
    axis : {"x", "y"}
        Streak direction on the image.

    Notes
    -----
    The unstreaked beam size in the streak direction limits the time
    resolution: ``sigma_t_min = sigma_unstreaked / S``.  Pass that size to
    :meth:`bunch_length` --- measure it by turning the deflector off.
    """

    screen: Screen
    shear: float
    axis: Literal["x", "y"] = "y"

    def __post_init__(self) -> None:
        if self.shear == 0:
            raise ValueError("shear must be non-zero; an unstreaked screen is a Screen")

    def measure(self, beam: Beam, rng: np.random.Generator | int | None = None) -> ScreenImage:
        """Image the beam.  Identical to ``self.screen.measure(beam, rng)``."""
        return self.screen.measure(beam, rng=rng)

    def _projection(self, image: ScreenImage):
        if self.axis == "x":
            return image.x_axis, image.projection_x(), image.counts.shape[0], 0
        return image.y_axis, image.projection_y(), image.counts.shape[1], 1

    def profile(self, image: ScreenImage) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Longitudinal profile: time in seconds against intensity in counts."""
        position, prof, rows, _ = self._projection(image)
        time = position / self.shear
        intensity = np.clip(prof - image.background * rows, 0.0, None)
        order = np.argsort(time)
        return time[order], intensity[order]

    def bunch_length(
        self, image: ScreenImage, unstreaked_size: float = 0.0, roi_sigma: float | None = 4.0
    ) -> float:
        """RMS bunch length in seconds.

        Parameters
        ----------
        image : ScreenImage
        unstreaked_size : float, optional
            RMS beam size in the streak direction with the deflector off,
            metres.  Removed in quadrature; this is the resolution limit.
        """
        position, prof, rows, index = self._projection(image)
        _, rms = profile_moments(position, prof, image.background * rows, roi_sigma)
        intrinsic = deconvolve(
            rms,
            unstreaked_size,
            image.psf_sigma[index],
            image.pixel_size[index] / np.sqrt(12),
        )
        return intrinsic / abs(self.shear)

    @property
    def resolution(self) -> float:
        """Best achievable time resolution in seconds, from the pixel pitch alone.

        The real limit is usually the unstreaked beam size, not this.
        """
        index = 0 if self.axis == "x" else 1
        return self.screen.beam_pixel_size[index] / abs(self.shear)
