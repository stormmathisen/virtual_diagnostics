# virtual_diagnostics

Turn simulated particle distributions into realistic accelerator diagnostic outputs.

Simulation stops at the beam. The control system starts at the signal. This package owns
the gap: give it a cloud of macroparticles in 6D phase space and it gives you the noisy
camera frame, the bandwidth-limited pickup waveform, or the charge-limited BPM reading
that the real instrument would have produced.

Built against [Cheetah](https://github.com/desy-ml/cheetah), but the internal beam is a
plain struct of NumPy arrays — any tracking code that can produce seven arrays plugs in.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[all]"     # or pick: cheetah, spice, hdf5, docs, dev
sudo apt install libngspice0          # only for the SPICE front end
```

Core dependencies are NumPy, SciPy and Matplotlib. Cheetah (and torch) is optional, and
so is PySpice.

## Sixty seconds

```python
import numpy as np
import virtual_diagnostics as vd

rng = np.random.default_rng(0)
n = 100_000
beam = vd.Beam(
    x=rng.normal(0, 150e-6, n), y=rng.normal(0, 80e-6, n),
    t=rng.normal(0, 1e-12, n),
    px=np.zeros(n), py=np.zeros(n), pz=np.full(n, 100e6),
    q=np.full(n, 250e-12 / n), energy=100e6,
)

screen = vd.Screen(pixel_size=10e-6, resolution=(400, 300), psf_sigma=25e-6, counts_per_pc=1e4)
image = screen.measure(beam, rng=0)

print(image.counts.shape, image.counts.dtype)   # (300, 400) uint16
print(image.beam_size(deconvolved=False))       # (152.17 um, 83.85 um)  raw
print(image.beam_size())                        # (150.08 um, 79.99 um)  PSF removed
print(beam.sigma_x, beam.sigma_y)               # (150.02 um, 80.18 um)  truth
```

The raw widths read high because the point spread adds in quadrature. That gap between
what the screen shows and what the beam is, is the point of the package.

## What is here

| Instrument | Class | Output |
| --- | --- | --- |
| Scintillator / OTR screen | `Screen` | 12- or 16-bit camera frame |
| Dispersive screen | `Spectrometer` | energy spectrum |
| Screen after a deflecting cavity | `StreakedScreen` | longitudinal profile |
| ICT, FCT, WCM, Faraday cup | `CurrentMonitor` | waveform, charge |
| Log-amp charge readout | `LogAmpReadout` | scalar PV |
| BPM, readout level | `BPM` | x, y, charge |
| Button BPM | `ButtonBPM` | one waveform per electrode |
| Stripline BPM | `StriplineBPM` | one waveform per electrode |
| CTR / CDR compression monitor | `CoherentRadiationMonitor` | scalar volts |
| Your signal conditioning | `SpiceFrontEnd` | whatever your netlist does |

Everything after the electrodes — delay lines, hybrids, filters, amplifiers — is
deliberately not a Python class. It goes in a SPICE netlist, and `SpiceFrontEnd` drives
that netlist with the diagnostic's signals, all sources in one simulation:

```python
combiner = vd.SpiceFrontEnd(
    netlist="examples/netlists/delay_line_combiner.cir",
    sources=("Vb1", "Vb2", "Vb3", "Vb4"),
    outputs="xout",
)
signals = bpm.measure(beam, rng=0)
t, trace = combiner.run(signals.t, signals.volts)
```

## Documentation

```bash
.venv/bin/mkdocs serve
```

- **Concepts** — the beam contract, units, sign conventions, calibration knobs
- **Guides** — screens, charge and current, BPMs, energy and bunch length, SPICE,
  other simulation codes
- **Physics and assumptions** — every model, and what it deliberately leaves out
- **API reference** — generated from the docstrings

## Example

```bash
.venv/bin/python examples/clara_diagnostics.py
```

Tracks a bunch through a short Cheetah lattice and reads it out on every diagnostic,
writing the figures used in the documentation.

## Tests

```bash
.venv/bin/pytest -q
```

Known-answer checks, not smoke tests: charge conservation through a screen, PSF adding in
quadrature, the Poisson kernel integrating to one, a stripline's zero DC response, the
log-amp round trip, BPM resolution scaling as 1/Q, and the Cheetah `tau`-to-time sign
convention.

## Licence

MIT.
