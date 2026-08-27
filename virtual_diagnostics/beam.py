"""The beam contract.

Everything in :mod:`virtual_diagnostics` consumes a :class:`Beam`: a plain
container of NumPy arrays in SI units.  It is deliberately a *struct*, not an
interface.  There is no adapter framework and no base class to subclass --- if
your tracking code can hand you seven arrays, it can drive every diagnostic in
this package via :func:`from_arrays`.

Conventions
-----------
======================  ==========================================================
``x``, ``y``            Transverse offsets from the reference trajectory, metres.
``t``                   Arrival time relative to the reference particle, seconds.
                        **Positive means the particle arrives later** (bunch tail).
``px``, ``py``, ``pz``  Momentum components in eV/c.
``q``                   Charge carried by each macroparticle, coulombs.
``energy``              Reference *total* energy of the bunch, eV.
======================  ==========================================================

Charges are stored with whatever sign the source uses.  Most tracking codes
(Cheetah included) report a positive magnitude even for electron bunches, so a
positive ``q`` is the normal case and the signals downstream come out positive.
The sign simply propagates linearly; nothing in this package takes an absolute
value of the current.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from scipy import constants

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike, NDArray

C_LIGHT = constants.speed_of_light
"""Speed of light in vacuum, m/s."""

REST_MASS_EV = {
    "electron": constants.physical_constants["electron mass energy equivalent in MeV"][0] * 1e6,
    "positron": constants.physical_constants["electron mass energy equivalent in MeV"][0] * 1e6,
    "proton": constants.physical_constants["proton mass energy equivalent in MeV"][0] * 1e6,
    "antiproton": constants.physical_constants["proton mass energy equivalent in MeV"][0] * 1e6,
    "deuteron": constants.physical_constants["deuteron mass energy equivalent in MeV"][0] * 1e6,
}
"""Rest energies in eV for the species this package knows by name."""


@dataclass
class Beam:
    """A cloud of macroparticles in 6D phase space.

    Parameters
    ----------
    x, y : ndarray
        Transverse offsets in metres, shape ``(n,)``.
    t : ndarray
        Arrival time relative to the reference particle in seconds, shape
        ``(n,)``.  Positive is later (bunch tail).
    px, py, pz : ndarray
        Momentum components in eV/c, shape ``(n,)``.
    q : ndarray
        Macroparticle charge in coulombs, shape ``(n,)``.
    energy : float
        Reference total energy of the bunch in eV.
    species : str, optional
        Particle species name.  Used only to look up the rest mass when
        ``rest_mass`` is not given explicitly.
    rest_mass : float, optional
        Rest energy in eV.  Overrides the ``species`` lookup, which is how you
        model a species this package has never heard of.

    Notes
    -----
    Statistical moments (``centroid_x``, ``sigma_t``, ...) are weighted by
    ``|q|`` so that they do not flip meaning if a code reports negative charges.
    :meth:`current_profile` uses the *signed* charge, because current has a
    direction.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 10_000
    >>> beam = Beam(
    ...     x=rng.normal(0, 1e-4, n), y=rng.normal(0, 1e-4, n),
    ...     t=rng.normal(0, 1e-12, n),
    ...     px=np.zeros(n), py=np.zeros(n), pz=np.full(n, 100e6),
    ...     q=np.full(n, 250e-12 / n), energy=100e6,
    ... )
    >>> round(beam.total_charge * 1e12, 3)
    250.0
    """

    x: NDArray[np.floating]
    y: NDArray[np.floating]
    t: NDArray[np.floating]
    px: NDArray[np.floating]
    py: NDArray[np.floating]
    pz: NDArray[np.floating]
    q: NDArray[np.floating]
    energy: float
    species: str = "electron"
    rest_mass: float | None = None

    def __post_init__(self) -> None:
        fields = ("x", "y", "t", "px", "py", "pz", "q")
        for name in fields:
            setattr(self, name, np.atleast_1d(np.asarray(getattr(self, name), dtype=float)))
        shapes = {name: getattr(self, name).shape for name in fields}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"Beam arrays must all have the same shape, got {shapes}")
        if getattr(self, "x").ndim != 1:
            raise ValueError(
                "Beam arrays must be 1-D. Vectorised (batched) beams are not "
                "supported; iterate over the batch and build one Beam per shot."
            )
        self.energy = float(self.energy)
        if self.rest_mass is None:
            try:
                self.rest_mass = REST_MASS_EV[self.species]
            except KeyError:
                raise ValueError(
                    f"Unknown species {self.species!r}; known species are "
                    f"{sorted(REST_MASS_EV)}. Pass rest_mass=<eV> explicitly for "
                    "anything else."
                ) from None
        self.rest_mass = float(self.rest_mass)

    # -- basic quantities -------------------------------------------------

    @property
    def n(self) -> int:
        """Number of macroparticles."""
        return self.x.size

    @property
    def total_charge(self) -> float:
        """Bunch charge in coulombs (signed, summed over macroparticles)."""
        return float(self.q.sum())

    @property
    def weights(self) -> NDArray[np.floating]:
        """Charge magnitudes, used to weight every statistical moment."""
        return np.abs(self.q)

    def _mean(self, a: NDArray[np.floating]) -> float:
        return float(np.average(a, weights=self.weights))

    def _std(self, a: NDArray[np.floating]) -> float:
        return float(np.sqrt(np.average((a - self._mean(a)) ** 2, weights=self.weights)))

    @property
    def centroid_x(self) -> float:
        """Charge-weighted mean horizontal position, m."""
        return self._mean(self.x)

    @property
    def centroid_y(self) -> float:
        """Charge-weighted mean vertical position, m."""
        return self._mean(self.y)

    @property
    def centroid_t(self) -> float:
        """Charge-weighted mean arrival time, s."""
        return self._mean(self.t)

    @property
    def sigma_x(self) -> float:
        """RMS horizontal beam size, m."""
        return self._std(self.x)

    @property
    def sigma_y(self) -> float:
        """RMS vertical beam size, m."""
        return self._std(self.y)

    @property
    def sigma_t(self) -> float:
        """RMS bunch length in time, s."""
        return self._std(self.t)

    @property
    def sigma_z(self) -> float:
        """RMS bunch length in space, m (``sigma_t`` times ``beta * c``)."""
        return self.sigma_t * self.relativistic_beta * C_LIGHT

    # -- energy -----------------------------------------------------------

    @property
    def energies(self) -> NDArray[np.floating]:
        """Total energy of every macroparticle, eV."""
        return np.sqrt(self.px**2 + self.py**2 + self.pz**2 + self.rest_mass**2)

    @property
    def mean_energy(self) -> float:
        """Charge-weighted mean total energy, eV."""
        return self._mean(self.energies)

    @property
    def energy_spread(self) -> float:
        """RMS absolute energy spread, eV."""
        return self._std(self.energies)

    @property
    def relative_energy_spread(self) -> float:
        """RMS energy spread divided by the mean energy (dimensionless)."""
        return self.energy_spread / self.mean_energy

    @property
    def relativistic_gamma(self) -> float:
        """Lorentz factor of the reference particle."""
        return self.energy / self.rest_mass

    @property
    def relativistic_beta(self) -> float:
        """Velocity of the reference particle as a fraction of ``c``."""
        gamma = self.relativistic_gamma
        return float(np.sqrt(1.0 - 1.0 / gamma**2))

    # -- longitudinal profile ---------------------------------------------

    def current_profile(
        self,
        bins: int = 200,
        dt: float | None = None,
        t_range: tuple[float, float] | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Bin the bunch into a beam current waveform.

        This is the single place longitudinal binning happens.  Both
        :class:`~virtual_diagnostics.electronic.CurrentMonitor` and
        :class:`~virtual_diagnostics.electronic.CoherentRadiationMonitor` call
        it rather than histogramming themselves.

        Parameters
        ----------
        bins : int, optional
            Number of time bins.  Ignored when ``dt`` is given.
        dt : float, optional
            Bin width in seconds.  Use this when the waveform has to line up
            with a digitiser sample rate.
        t_range : tuple of float, optional
            ``(t_min, t_max)`` in seconds.  Defaults to +/- 5 sigma about the
            bunch centroid, which keeps the tails without wasting bins.

        Returns
        -------
        t : ndarray
            Bin *centres* in seconds, uniformly spaced.
        current : ndarray
            Beam current in amperes, ``charge_in_bin / dt``.

        Notes
        -----
        The returned current is signed exactly as ``q`` is.  A profile binned
        this way is a histogram, so it carries shot noise from the finite
        macroparticle count; use enough macroparticles that this is below the
        instrument noise you care about.
        """
        if t_range is None:
            centre, sigma = self.centroid_t, self.sigma_t
            span = 5.0 * sigma if sigma > 0 else 1e-12
            t_range = (centre - span, centre + span)
        lo, hi = float(t_range[0]), float(t_range[1])
        if not hi > lo:
            raise ValueError(f"t_range must be increasing, got {t_range}")
        if dt is not None:
            if dt <= 0:
                raise ValueError(f"dt must be positive, got {dt}")
            bins = max(int(np.ceil((hi - lo) / dt)), 1)
            hi = lo + bins * dt
        charge, edges = np.histogram(self.t, bins=bins, range=(lo, hi), weights=self.q)
        width = edges[1] - edges[0]
        return 0.5 * (edges[:-1] + edges[1:]), charge / width

    # -- manipulation ------------------------------------------------------

    def shifted(self, dx: float = 0.0, dy: float = 0.0, dt: float = 0.0) -> Beam:
        """Return a copy displaced in ``x``, ``y`` and ``t``.

        Handy for shot-to-shot jitter studies without touching the tracker.
        """
        return replace(self, x=self.x + dx, y=self.y + dy, t=self.t + dt)

    def scaled_charge(self, total_charge: float) -> Beam:
        """Return a copy rescaled to a given total charge in coulombs."""
        return replace(self, q=self.q * (total_charge / self.total_charge))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Beam(n={self.n}, charge={self.total_charge * 1e12:.3g} pC, "
            f"energy={self.energy / 1e6:.4g} MeV, species={self.species!r}, "
            f"sigma_x={self.sigma_x * 1e6:.3g} um, sigma_y={self.sigma_y * 1e6:.3g} um, "
            f"sigma_t={self.sigma_t * 1e15:.3g} fs)"
        )


# -- adapters -------------------------------------------------------------


def from_arrays(
    x: ArrayLike,
    y: ArrayLike,
    t: ArrayLike,
    px: ArrayLike,
    py: ArrayLike,
    pz: ArrayLike,
    q: ArrayLike,
    energy: float,
    species: str = "electron",
    rest_mass: float | None = None,
) -> Beam:
    """Build a :class:`Beam` from raw arrays.

    This is the integration point for every tracking code that is not Cheetah.
    There is no plugin system to register with: convert your distribution to
    the units in the module docstring and call this.

    Examples
    --------
    ASTRA-style arrays, where ``z`` is the longitudinal offset in metres and a
    particle ahead of the reference (``z > 0``) arrives *earlier*::

        beam = from_arrays(
            x=x, y=y, t=-z / (beta * C_LIGHT),
            px=px, py=py, pz=pz, q=charges, energy=E0,
        )
    """
    return Beam(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        t=np.asarray(t, dtype=float),
        px=np.asarray(px, dtype=float),
        py=np.asarray(py, dtype=float),
        pz=np.asarray(pz, dtype=float),
        q=np.asarray(q, dtype=float),
        energy=energy,
        species=species,
        rest_mass=rest_mass,
    )


def from_cheetah(particle_beam) -> Beam:
    """Convert a :class:`cheetah.ParticleBeam` into a :class:`Beam`.

    Momenta come from Cheetah's own ``to_xyz_pxpypz()`` (kg m/s, converted to
    eV/c here) rather than being re-derived from the normalised phase-space
    columns, so this stays correct if Cheetah changes its internal convention.

    Arrival time is taken from the ``tau`` coordinate as ``t = tau / c``.
    Cheetah defines the longitudinal position as ``z = -beta * tau``, and a
    particle ahead of the reference (``z > 0``) crosses a downstream plane
    earlier, so ``t = -z / (beta c) = tau / c``.  The sign is pinned by
    ``tests/test_beam.py::test_cheetah_head_arrives_first``.

    Parameters
    ----------
    particle_beam : cheetah.ParticleBeam
        An unbatched beam.  Cheetah supports a leading vector dimension for
        parallel tracking; slice it into individual beams first.

    Raises
    ------
    ValueError
        If the beam carries a vector (batch) dimension.

    Notes
    -----
    Lost particles are folded in through ``survival_probabilities``, which
    scales each macroparticle charge rather than removing the particle.  A
    fully lost particle therefore contributes zero charge and drops out of
    every charge-weighted moment on its own.
    """
    particles = particle_beam.particles
    if particles.ndim > 2:
        raise ValueError(
            f"Expected an unbatched ParticleBeam, got particles with shape "
            f"{tuple(particles.shape)}. Slice the vector dimension first, e.g. "
            "from_cheetah(beam[i])."
        )

    def np_(tensor):
        return np.asarray(tensor.detach().cpu().numpy(), dtype=float)

    si = np_(particle_beam.to_xyz_pxpypz())
    # kg m/s -> eV/c
    to_ev_c = C_LIGHT / constants.elementary_charge
    charges = np_(particle_beam.particle_charges)
    survival = np_(particle_beam.survival_probabilities)

    return Beam(
        x=si[..., 0],
        y=si[..., 2],
        t=np_(particles[..., 4]) / C_LIGHT,
        px=si[..., 1] * to_ev_c,
        py=si[..., 3] * to_ev_c,
        pz=si[..., 5] * to_ev_c,
        q=charges * survival,
        energy=float(np_(particle_beam.energy)),
        species=particle_beam.species.name,
        rest_mass=float(np_(particle_beam.species.mass_eV)),
    )


def from_astra(path: str) -> Beam:
    """Load an ASTRA particle file, via Cheetah's reader.

    Requires the ``cheetah`` extra.  This is a two-line delegation, not a
    reimplementation --- if you need ASTRA support without torch installed,
    read the file yourself and call :func:`from_arrays`.
    """
    from cheetah import ParticleBeam

    return from_cheetah(ParticleBeam.from_astra(path))


def from_openpmd(path: str, energy: float) -> Beam:
    """Load an openPMD-beamphysics HDF5 file, via Cheetah's reader.

    Requires the ``cheetah`` extra.  ``energy`` is the reference energy in eV,
    which openPMD does not itself define.
    """
    from cheetah import ParticleBeam

    return from_cheetah(ParticleBeam.from_openpmd_file(path, energy=energy))
