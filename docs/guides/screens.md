# Screens and cameras

A [`Screen`](../api/screen.md) turns the transverse charge density into a camera
frame. Everything between the charge and the ADC code is lumped into knobs you can
calibrate against a real device, because that is how screens are characterised in a
control room: a point source for the PSF, and total image counts against an ICT reading
for the sensitivity.

## The pipeline

1. Bin the charge onto the pixel grid.
2. Blur with the Gaussian point spread function.
3. Convert to counts through `counts_per_pc`.
4. Add photon shot noise, on photoelectrons, scaled by `gain`.
5. Add the pedestal and read noise.
6. Clip and quantise to `bits`.

Blurring happens *before* the noise because the PSF is optical and the noise is
detection. Doing it the other way round smooths the noise and gives a falsely clean
image.

## A first measurement

```python
import numpy as np
import virtual_diagnostics as vd

rng = np.random.default_rng(0)
n = 100_000
beam = vd.Beam(
    x=rng.normal(0, 150e-6, n), y=rng.normal(0, 80e-6, n), t=rng.normal(0, 1e-12, n),
    px=np.zeros(n), py=np.zeros(n), pz=np.full(n, 100e6),
    q=np.full(n, 250e-12 / n), energy=100e6,
)

screen = vd.Screen(pixel_size=10e-6, resolution=(400, 300), psf_sigma=25e-6, counts_per_pc=1e4)
image = screen.measure(beam, rng=0)

print("frame:", image.counts.shape, image.counts.dtype, "peak", image.counts.max())
print("raw sigma_x/y (um):", tuple(round(v * 1e6, 2) for v in image.beam_size(deconvolved=False)))
print("deconvolved (um): ", tuple(round(v * 1e6, 2) for v in image.beam_size()))
print("true (um):        ", (round(beam.sigma_x * 1e6, 2), round(beam.sigma_y * 1e6, 2)))
```

```text
frame: (300, 400) uint16 peak 3280
raw sigma_x/y (um): (152.17, 83.85)
deconvolved (um):  (150.08, 79.99)
true (um):         (150.02, 80.18)
```

![Screen image and projections](../images/screen.png)

## Resolution: what a screen can and cannot see

The measured width is the beam and the PSF and the pixel pitch added in quadrature.
`beam_size()` removes the last two; what is left is the beam, until the beam is small
enough that nothing is left.

```python
for sigma in (200e-6, 100e-6, 50e-6, 20e-6, 5e-6):
    beam = gaussian(sigma_x=sigma, sigma_y=sigma)
    screen = vd.Screen(pixel_size=10e-6, resolution=(400, 300), psf_sigma=60e-6,
                       counts_per_pc=1e3, bits=16)
    image = screen.measure(beam, rng=0)
    print(sigma * 1e6, image.beam_size(deconvolved=False)[0] * 1e6, image.beam_size()[0] * 1e6)
```

```text
  true    raw   deconvolved   quadrature expects raw
 200.0  209.10     200.28     208.83
 100.0  117.68     101.20     116.65
  50.0   78.81      51.01      78.16
  20.0   62.49      17.23      63.31
   5.0   59.97       0.00      60.28
```

A 60 µm PSF measures a 200 µm beam to a fraction of a percent, a 50 µm beam to about
2 %, and a 5 µm beam not at all: the raw width is just the PSF, and the deconvolution
correctly collapses to zero. **Zero means unresolved, not zero width.**

!!! tip "Measure the PSF, do not guess it"
    `psf_sigma` is the single number that decides whether a measurement is real. On a
    thick YAG screen it is dominated by the scintillator, not the optics. Calibrate it
    against a beam you have measured another way, or against a resolution target.

## Saturation

Saturation is the most common way a screen measurement goes quietly wrong. A flattened
core widens the profile, so the beam reads *large*:

```python
for cpp in (3e3, 1e4, 3e4, 1e5):
    screen = vd.Screen(pixel_size=12e-6, resolution=(640, 480), psf_sigma=30e-6,
                       counts_per_pc=cpp, bits=12)
    image = screen.measure(beam, rng=0)          # true sigma_x = 46.0 um
    print(cpp, image.counts.max(), image.saturated_fraction, image.beam_size()[0])
```

```text
  counts_per_pc=   3000  peak= 2919  saturated= 0.000%  sigma_x= 45.59 um
  counts_per_pc=  10000  peak= 4095  saturated= 0.075%  sigma_x= 51.70 um
  counts_per_pc=  30000  peak= 4095  saturated= 0.173%  sigma_x= 63.53 um
  counts_per_pc= 100000  peak= 4095  saturated= 0.278%  sigma_x= 75.60 um
```

Less than a *tenth of a percent* of pixels at the top code already inflates the width by
12 %. Always check `image.saturated_fraction` before trusting a size.

## Charge that misses the sensor

```python
beam = gaussian(centroid_x=2.5e-3)                # sensor is only +/- 2 mm
image = vd.Screen(pixel_size=10e-6, resolution=(400, 300)).measure(beam, rng=0)
print(image.off_screen_fraction)                  # 0.9995
```

`off_screen_fraction` is the fraction of bunch charge that never reached a pixel. Any
value above a percent or so means the measured widths are truncated.

## Taking moments without lying to yourself

`ScreenImage.moments_x` and `moments_y` subtract the pedestal, locate the beam, and take
**signed** moments inside a region of interest. Both halves of that matter:

- Rectifying the background-subtracted residual at zero gives every empty bin a small
  positive weight. The second moment weights bins by $(x-\mu)^2$, so the tails dominate
  and the width comes out roughly 15 % high.
- Keeping the residual signed but using the whole axis is unbiased but wildly noisy on a
  sensor much larger than the beam — it can return a negative variance and a NaN.

So the region of interest is located by an iterated clipped estimate and the final
moments are taken signed inside it. On a 640×480 sensor with a 46 µm beam, that is the
difference between a stable 46 µm and a useless one:

```python
sizes = [screen.measure(beam, rng=seed).beam_size() for seed in range(10)]
# with the ROI:    46.0 um, scatter 0.3 um
# roi_sigma=None: unstable, occasionally NaN
```

Pass `roi_sigma=None` to disable it, or a different number of widths to widen or tighten
the window.

## Screen tilt

A screen set at 45° to the beam is longer in the horizontal direction than it looks, so
one pixel subtends less beam:

```python
screen = vd.Screen(pixel_size=12e-6, tilt=np.pi / 4)
screen.beam_pixel_size    # (8.485e-06, 1.2e-05)
```

Axes returned in `ScreenImage` are already in **beam** coordinates, so widths compare
directly against `beam.sigma_x`. If your camera views along the beam axis the factor has
already cancelled — leave `tilt` at zero and put everything into `pixel_size`.

## Plotting

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
vd.plot_image(image, ax=axes[0])
vd.plot_projections(image, ax=axes[1])
```

Both take `ax=`, both return the axis, and `plot_image` accepts any `imshow` keyword.
