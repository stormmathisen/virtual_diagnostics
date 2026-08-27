# Other simulation codes

There is no adapter framework here, no plugin registry, and nothing to subclass. The
integration point is [`from_arrays`](../api/beam.md): convert your distribution to the
units below and call it.

| Quantity | Unit | Convention |
| --- | --- | --- |
| `x`, `y` | metres | offset from the reference trajectory |
| `t` | seconds | relative to the reference particle, **positive = arrives later** |
| `px`, `py`, `pz` | eV/c | not normalised to a reference momentum |
| `q` | coulombs | charge of each *macroparticle* |
| `energy` | eV | reference **total** energy |

## The one thing to get right

Most tracking codes carry a longitudinal *position* `z`, where a particle ahead of the
reference has `z > 0`. Ahead in space means **earlier** in time at a fixed downstream
plane, so the conversion carries a minus sign:

```python
t = -z / (beta * vd.C_LIGHT)
```

```python
z = rng.normal(0, 300e-6, 5000)          # metres, +z is ahead
gamma = 100e6 / 510998.95
beta = np.sqrt(1 - 1 / gamma**2)

beam = vd.from_arrays(
    x=x, y=y,
    t=-z / (beta * vd.C_LIGHT),
    px=px, py=py, pz=pz,
    q=np.full(5000, 250e-12 / 5000),
    energy=100e6,
)
print(beam)
print(beam.sigma_z * 1e6, "um")
```

```text
Beam(n=5000, charge=250 pC, energy=100 MeV, species='electron', sigma_x=99.7 um, sigma_y=100 um, sigma_t=1e+03 fs)
  sigma_z back out (um): 300.47  (input 300.0 um)
```

Get the sign backwards and every longitudinal measurement is time-reversed — silently,
and symmetrically enough that nothing looks wrong. **Write a test that pins it.** Build a
distribution with a deliberate head–tail asymmetry and assert that the head arrives
first; `tests/test_cheetah.py::test_head_arrives_first` does exactly that for the Cheetah
adapter.

## Momentum units

If your code gives momenta in kg·m/s, multiply by $c/e$:

```python
px_ev_c = px_si * vd.C_LIGHT / 1.602176634e-19
```

If it gives normalised momenta $p_x/p_0$ (Bmad-style, as Cheetah does internally),
multiply by the reference momentum in eV/c. If it gives a fractional momentum deviation
$\delta$, the total momentum is $p_0(1+\delta)$ and

```python
pz = np.sqrt((p0 * (1 + delta))**2 - px**2 - py**2)
```

## Species other than electrons

Named species (`electron`, `positron`, `proton`, `antiproton`, `deuteron`) look up their
own rest mass. Anything else needs one:

```python
vd.from_arrays(..., species="muon", rest_mass=105.66e6)
```

## Charge sign

Store whatever sign your code uses. Most report a positive magnitude even for electrons,
so signals come out positive; the sign propagates linearly and nothing here takes an
absolute value of the current. Statistical moments weight by `|q|`, so a code that
reports negative charges still gets a sensible beam size.

## Cheetah

Cheetah has a real adapter because it is the reference source:

```python
beam = vd.from_cheetah(particle_beam)
```

It uses Cheetah's own `to_xyz_pxpypz()` for the momenta rather than re-deriving them, so
it stays correct if Cheetah changes its internal convention. Lost particles are folded in
through `survival_probabilities`, which scales each macroparticle charge — a fully lost
particle contributes zero charge and drops out of every moment on its own. Vectorised
(batched) beams are rejected with a clear error; slice them first.

`from_astra` and `from_openpmd` delegate to Cheetah's readers, so they need the `cheetah`
extra. If you want those formats without torch installed, read the file yourself and call
`from_arrays`.

## A checklist for a new source

1. Convert positions to metres, momenta to eV/c, charges to coulombs.
2. Convert the longitudinal coordinate to arrival time, minus sign included.
3. Check `beam.sigma_z` against the bunch length you expect.
4. Check `beam.mean_energy` against your reference energy.
5. Write the head-arrives-first test.
