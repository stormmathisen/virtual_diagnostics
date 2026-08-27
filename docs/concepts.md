# Concepts

## The `Beam` is a struct, not an interface

Everything in this package consumes a [`Beam`](api/beam.md): a plain container of NumPy arrays
in SI units. There is no base class to subclass and nothing to register.

```python
@dataclass
class Beam:
    x: np.ndarray    # m
    y: np.ndarray    # m
    t: np.ndarray    # s, relative to the reference particle; +ve arrives later
    px: np.ndarray   # eV/c
    py: np.ndarray   # eV/c
    pz: np.ndarray   # eV/c
    q:  np.ndarray   # C, per macroparticle
    energy: float    # eV, reference total energy
    species: str = "electron"
    rest_mass: float | None = None   # eV; overrides the species lookup
```

This is the whole integration story. A tracking code that can hand you seven equal-length
arrays can drive every diagnostic here — see [Other simulation codes](guides/other-codes.md).

### Units and sign conventions

| Quantity | Unit | Convention |
| --- | --- | --- |
| `x`, `y` | metres | offset from the reference trajectory |
| `t` | seconds | relative to the reference particle, **positive = arrives later** |
| `px`, `py`, `pz` | eV/c | not normalised to a reference momentum |
| `q` | coulombs | charge of each *macroparticle* |
| `energy` | eV | reference **total** energy, not kinetic |

!!! warning "The `t` sign is the one that bites"
    Most tracking codes carry a longitudinal *position* `z`, where a particle ahead of the
    reference has `z > 0`. Ahead in space means **earlier** in time at a fixed downstream plane,
    so the conversion carries a minus sign:

    ```python
    t = -z / (beta * C_LIGHT)
    ```

    Get it backwards and every longitudinal measurement is time-reversed, silently and
    symmetrically — nothing looks wrong. The Cheetah adapter's convention is pinned by
    `tests/test_cheetah.py::test_head_arrives_first`; do the same for your own adapter.

Charges are stored with whatever sign your source uses. Most codes report a positive magnitude
even for electrons, so signals come out positive. The sign propagates linearly and nothing in
the package takes an absolute value of the current. Statistical moments weight by `|q|`, so a
code that reports negative charges still gets a sensible beam size.

## Derived quantities

Everything you would otherwise recompute is already a property:

```python
for name in ("total_charge", "n", "sigma_x", "sigma_t", "mean_energy",
             "relative_energy_spread", "relativistic_gamma", "sigma_z"):
    print(f"beam.{name:<24} = {getattr(beam, name):.6g}")
```

```text
beam.total_charge             = 2.5e-10
beam.n                        = 100000
beam.sigma_x                  = 0.000150019
beam.sigma_t                  = 1.00094e-12
beam.mean_energy              = 1.00001e+08
beam.relative_energy_spread   = 0
beam.relativistic_gamma       = 195.695
beam.sigma_z                  = 0.00030007
```

Also available: `centroid_x`, `centroid_y`, `centroid_t`, `sigma_y`, `energies` (per particle),
`energy_spread`, `relativistic_beta`, `weights`.

Two cheap manipulations avoid a round trip through the tracker:

```python
print(beam.shifted(dx=1e-3).centroid_x * 1e6)      # 999.9  (um)
print(beam.scaled_charge(10e-12).total_charge)     # 1e-11
```

`shifted` is for shot-to-shot jitter studies; `scaled_charge` is for charge scans like the one
in the [BPM guide](guides/bpm.md). Both return copies.

## The longitudinal profile is computed in one place

`Beam.current_profile` is the single place longitudinal binning happens. Both `CurrentMonitor`
and `CoherentRadiationMonitor` call it rather than histogramming themselves.

```python
t, current = beam.current_profile(bins=9)
print("t (ps): ", np.round(t * 1e12, 3))
print("I (A):  ", np.round(current, 1))
print("integral (pC):", round(float((current * (t[1] - t[0])).sum()) * 1e12, 3))
```

```text
t (ps):  [-4.448 -3.336 -2.224 -1.111  0.001  1.113  2.225  3.337  4.449]
I (A):   [ 0.   0.6 10.2 53.9 94.8 54.6 10.   0.6  0. ]
integral (pC): 250.0
```

It returns bin **centres** and a current in amperes, and it conserves charge. Pass `dt=` instead
of `bins=` when the waveform has to line up with a digitiser sample rate; pass `t_range=` to
override the default window of ±5σ about the bunch centroid.

!!! note "Macroparticle count sets a noise floor"
    A binned profile is a histogram, so it carries shot noise from the finite macroparticle
    count. That floor matters most for the [form factor](guides/longitudinal.md#compression-coherent-radiation),
    where it limits how deep into the tail of `|F(f)|²` you can trust the result — roughly
    `1/N`. Use enough macroparticles that this sits below the instrument noise you care about.

## Noise, and why every diagnostic takes an `rng`

Every `measure()` accepts `rng`: a `numpy.random.Generator`, an integer seed, or `None` for
fresh entropy. Pass an integer and the shot is reproducible; pass the *same generator instance*
to every diagnostic in a shot and the whole machine replays identically.

```python
image_a = screen.measure(beam, rng=7)
image_b = screen.measure(beam, rng=7)
assert np.array_equal(image_a.counts, image_b.counts)
```

The noise primitives themselves live in [`virtual_diagnostics.noise`](api/noise.md) as free
functions — `shot`, `read`, `quantise`, `jitter`. There is no noise class hierarchy to
configure. Each instrument exposes the noise parameters a real datasheet quotes, and nothing
else.

## Calibration knobs are the point

Hardware is never the ideal on paper. A screen's light yield drifts with dose, an ICT's
constants are refitted against a Faraday cup, a BPM's electrical centre is not its magnetic
centre. Every instrument here therefore exposes the same knobs you would calibrate on the real
device, rather than deriving everything from first principles:

- `Screen.counts_per_pc` — one lumped number for light yield, optics, quantum efficiency and
  camera gain, because that product is what you measure against an ICT.
- `LogAmpReadout.qcal` / `.ucal` — refit them from cup data and drop them straight in.
- `BPM.gain` / `.offset` — scale error and electrical centre offset.
- `Spectrometer.dispersion`, `StreakedScreen.shear` — the calibration you measure by scanning.

## Where the physics stops

`Spectrometer` and `StreakedScreen` contain no new physics. The dipole and the transverse
deflecting cavity are your tracking code's job: track the beam to the screen, then use these
classes to turn pixel positions into energy or time. See
[Energy and bunch length](guides/longitudinal.md) for the worked recipe, and
[Physics and assumptions](physics.md) for what every model leaves out.
