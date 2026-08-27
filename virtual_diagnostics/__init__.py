"""Turn simulated particle distributions into realistic diagnostic outputs.

Simulation stops at the beam; the control system starts at the signal.  This
package covers the gap: give it a cloud of macroparticles and it gives you the
noisy camera frame, the bandwidth-limited pickup waveform or the charge-limited
BPM reading that the real instrument would have produced.

Quick start
-----------
>>> import numpy as np
>>> from virtual_diagnostics import Beam, Screen
>>> rng = np.random.default_rng(0)
>>> n = 10_000
>>> beam = Beam(
...     x=rng.normal(0, 150e-6, n), y=rng.normal(0, 80e-6, n),
...     t=rng.normal(0, 1e-12, n),
...     px=np.zeros(n), py=np.zeros(n), pz=np.full(n, 100e6),
...     q=np.full(n, 250e-12 / n), energy=100e6,
... )
>>> image = Screen(pixel_size=10e-6, resolution=(200, 150)).measure(beam, rng=0)
>>> image.counts.shape
(150, 200)

See :mod:`virtual_diagnostics.beam` for the beam contract and how to feed in
distributions from tracking codes other than Cheetah.
"""

from .beam import C_LIGHT, Beam, from_arrays, from_astra, from_cheetah, from_openpmd
from .bpm import (
    BPM,
    DIAGONAL_ELECTRODES,
    BpmReading,
    ButtonBPM,
    ButtonSignals,
    ElectrodePickup,
    ElectrodeSignals,
    StriplineBPM,
    wall_current_fraction,
)
from .electronic import CoherentRadiationMonitor, CurrentMonitor, LogAmpReadout
from .export import to_csv, to_hdf5, to_spice_pwl
from .noise import default_rng, jitter, quantise, read, shot
from .plotting import plot_image, plot_projections, plot_signal
from .screen import Screen, ScreenImage, Spectrometer, StreakedScreen, deconvolve, profile_moments
from .spice import (
    NgspiceError,
    NgspiceNotFound,
    SpiceFrontEnd,
    ngspice_available,
    ngspice_executable,
    ngspice_version,
    read_rawfile,
)

__version__ = "0.1.0"

__all__ = [
    "BPM",
    "Beam",
    "BpmReading",
    "ButtonBPM",
    "ButtonSignals",
    "C_LIGHT",
    "CoherentRadiationMonitor",
    "CurrentMonitor",
    "DIAGONAL_ELECTRODES",
    "ElectrodePickup",
    "ElectrodeSignals",
    "LogAmpReadout",
    "NgspiceError",
    "NgspiceNotFound",
    "Screen",
    "ScreenImage",
    "Spectrometer",
    "SpiceFrontEnd",
    "StriplineBPM",
    "StreakedScreen",
    "__version__",
    "deconvolve",
    "default_rng",
    "from_arrays",
    "from_astra",
    "from_cheetah",
    "from_openpmd",
    "jitter",
    "plot_image",
    "plot_projections",
    "plot_signal",
    "profile_moments",
    "quantise",
    "read_rawfile",
    "ngspice_available",
    "ngspice_executable",
    "ngspice_version",
    "read",
    "shot",
    "to_csv",
    "to_hdf5",
    "to_spice_pwl",
    "wall_current_fraction",
]
