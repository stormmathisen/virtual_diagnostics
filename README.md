# virtual_diagnostics

[![CI](https://github.com/stormmathisen/virtual_diagnostics/actions/workflows/ci.yml/badge.svg)](https://github.com/stormmathisen/virtual_diagnostics/actions/workflows/ci.yml)

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
.venv/bin/pip install -e ".[all]"     # or pick: cheetah, hdf5, docs, dev
sudo apt install ngspice              # only for the SPICE front end
```

Core dependencies are NumPy, SciPy and Matplotlib. Cheetah (and torch) is optional. The
SPICE front end shells out to the **ngspice binary** — no Python SPICE binding is used or
needed; without it, `ngspice_available()` is `False` and everything else still works.

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

Optional integrations skip when their dependency is missing, which is right locally and
dangerous in CI — a runner without ngspice or Cheetah reports success while a fifth of the
suite never runs. Set `VD_REQUIRE_ALL_EXTRAS=1` to turn a missing optional dependency into
a failure instead:

```bash
VD_REQUIRE_ALL_EXTRAS=1 .venv/bin/pytest -q -ra
```

## CI

Three jobs, on push and pull request:

- **core only** — installs no optional dependency at all, and asserts that torch, Cheetah,
  h5py and ngspice are absent. This is what makes the "core needs only NumPy, SciPy and
  Matplotlib" claim true, and it checks that `NgspiceNotFound` still carries installation
  instructions.
- **full** — Python 3.10 and 3.12, with ngspice and every extra installed and
  `VD_REQUIRE_ALL_EXTRAS=1` so nothing can skip silently. Also runs the example end to end.
- **docs** — `mkdocs build --strict`, uploading the built site as an artifact.
- **publish docs** — dormant. See below.

### Reading the docs from CI

```bash
gh run download -n site -D site && python -m http.server -d site 8000
```

### Publishing the docs

A `publish-docs` job is wired up but disabled. GitHub Pages **cannot serve a private
repository on the Free plan**, and on Pro/Team a Pages site built from a private repo is
publicly readable — only Enterprise Cloud offers access-controlled private Pages. So
publishing means either upgrading, or making the repo public.

When either is true:

```bash
gh api -X POST repos/stormmathisen/virtual_diagnostics/pages -f build_type=workflow
gh variable set PUBLISH_DOCS --body true
```

The next push to `main` deploys to <https://stormmathisen.github.io/virtual_diagnostics/>.

Known-answer checks, not smoke tests: charge conservation through a screen, PSF adding in
quadrature, the Poisson kernel integrating to one, a stripline's zero DC response, the
log-amp round trip, BPM resolution scaling as 1/Q, and the Cheetah `tau`-to-time sign
convention.

## Licence

MIT.
