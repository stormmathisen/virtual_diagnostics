# Physics and assumptions

What each model does, and — more usefully — what it does not.

## Screens

**Model.** Charge is binned onto the pixel grid, blurred with a Gaussian point spread
function, converted to counts through one lumped sensitivity, given photon shot noise on
photoelectrons, then a pedestal, read noise, clipping and quantisation.

**Why lumped.** Light yield, collection solid angle, quantum efficiency and camera gain
only ever appear as a product, and that product is what you calibrate against an ICT.
Splitting it into factors you cannot measure separately would be false precision.

**Assumes.** The PSF is Gaussian and spatially uniform. The response is linear up to
saturation. The screen is thin enough that depth of field does not matter.

**Does not model.** Scintillator saturation at high dose (a real nonlinearity on YAG at
high charge density), afterglow between shots, camera blooming, OTR's angular emission
pattern and its coherent enhancement for short bunches, radiation damage, or the vacuum
window.

**Consequence.** Trust widths when `saturated_fraction` is zero and
`off_screen_fraction` is small. A beam much smaller than `psf_sigma` is unresolved and
the deconvolution returns zero, which means *unresolved*, not zero width.

## Taking moments from an image

The estimator subtracts the pedestal, locates the beam with iterated clipped estimates,
and takes **signed** moments inside a region of interest. Both choices are load-bearing:

- Rectifying the background-subtracted residual at zero gives every empty bin a small
  positive weight. Since the second moment weights bins by $(x-\mu)^2$, the tails
  dominate and the width reads about 15 % high.
- Keeping the residual signed but using the whole axis is unbiased but very noisy on a
  sensor much larger than the beam, and can return a negative variance.

Truncating a Gaussian at four widths costs well under a percent, and the locator is
biased wide, which keeps the region generous. `roi_sigma=None` disables it.

## Current monitors

**Model.** Beam current binned at the digitiser rate, then a single-pole low pass (the
rise time) followed by a single-pole high pass (the droop), then transimpedance, noise
floor and optional quantisation.

**Assumes.** The response is linear and time-invariant, and two poles describe it. That
is the same two-parameter description every transformer datasheet uses.

**Does not model.** Ringing, reflections on the cable, saturation of the magnetic core,
the exact shape of a real transformer's rolloff, or common-mode pickup.

**Consequence.** Droop under-reads an integrated charge over a finite record, and the
size of that error is exactly what the model gives you. Absolute pulse height is only
right when the sample interval is well inside the rise time: a one-pole response sampled
at `dt` peaks at $Q/(\tau+dt)$, not $Q/\tau$.

## Log-amp readout

$Q = Q_\mathrm{Cal}\cdot 10^{V/U_\mathrm{Cal}}$, the model CLARA's ICT IOCs use, fitted
from a straight line of $\log_{10}Q$ against voltage. It is a *calibration*, not a
physical model of a logarithmic amplifier: no temperature drift, no dynamic-range limits,
no settling time.

## BPMs

### Readout model

Centroid plus Gaussian noise scaling as `reference_charge / charge`, with gain and offset
knobs. Linear by construction. Does not model nonlinearity, electrode geometry, or
anything in the time domain.

### Per-electrode models

**Model.** The wall image current density is the Poisson kernel for a beam inside a
circular pipe; each electrode intercepts its integral over its own angular width. The
electrode response is then either an $R\,C$ high pass (button) or the difference of two
delayed copies of the current (stripline).

**Exact, not approximate.** The kernel is evaluated at the bunch centroid. The Poisson
kernel is harmonic inside the pipe, so its average over any circularly symmetric charge
distribution equals its value at the centre — for a round beam this is exact. An
elliptical beam near the wall picks up a correction of order
$(\sigma_x^2-\sigma_y^2)/b^2$.

**Assumes.** A circular pipe, a perfectly conducting wall, an ultrarelativistic beam so
the field is transverse, and electrodes small enough that their own geometry does not
perturb the wall current.

**Does not model.** Non-circular chambers, the electrode's own field distortion, the
transition impedance at the electrode gap, cross-plane coupling, trapped modes, or
wakefields.

**Consequence.** The position readout goes nonlinear towards the wall — around 7 % low at
$0.3b$ and 30 % low at $0.7b$ — and that nonlinearity comes out of the geometry rather
than being imposed. Electrode gain mismatch turns straight into a position offset.

### Striplines

$V(t) = \frac{1}{2}Z_0 f_i [I(t) - I(t-\tau)]$ with $\tau = L/(\beta c) + L/c$. This gives
no DC response, a comb transfer function $2|\sin(\pi f\tau)|$ peaking at $c/4L$, and
directivity — all three of which are checked in the tests. Directivity is modelled as a
flat leakage in dB onto the far port, not as a frequency-dependent coupling.

**Does not model.** Line loss and dispersion, mismatch reflections at the feedthroughs,
or the electrode's finite width in the longitudinal direction.

## Coherent radiation

Signal $\propto Q^2\langle|F(f)|^2\rangle$ over the detector band, where $F$ is the
normalised Fourier transform of the longitudinal profile. The single-particle emission
spectrum is taken as flat across the band; diffraction, detector response and transport
optics all fold into one calibration constant.

**Consequence.** The *scaling* with bunch length is right, which is what a compression
scan needs. The absolute power is not. The form factor is limited by the macroparticle
count to roughly $1/N$, so a detector band sitting where $|F|^2$ is very small needs a
lot of particles.

## Spectrometers and deflecting cavities

No physics at all: these are a `Screen` plus an axis calibration. The dipole and the
cavity belong in your tracking code, which models them properly. What these classes add
is the calibration and the quadrature subtraction of the contributions that are not what
you are trying to measure — the betatron beam size for an energy spread, the unstreaked
size for a bunch length.

## Noise

Poisson counting statistics, switching to a Gaussian of matched mean and variance above a
million quanta; zero-mean Gaussian read noise; clipping and rounding onto an ADC ladder.
No fixed-pattern noise, no $1/f$, no correlated pixel noise, no bit errors.

## Not modelled anywhere

Space charge, wakefields, CSR, and every other collective effect: those belong upstream,
in the tracking code. Electro-optic sampling. Beam loss monitors. Emittance measurement
as such — do a quadrupole scan with `Screen` and fit it yourself. Multi-bunch and
turn-by-turn behaviour: everything here is single-shot.
