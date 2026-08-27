"""Screens: charge conservation, resolution, saturation, and the two wrappers."""

import numpy as np
import pytest
from conftest import SIGMA_T, SIGMA_X, SIGMA_Y, TOTAL_CHARGE, gaussian_beam

from virtual_diagnostics import Screen, Spectrometer, StreakedScreen
from virtual_diagnostics.screen import deconvolve, profile_moments


def quiet_screen(**overrides):
    """A screen with no pedestal or read noise, for known-answer tests."""
    settings = dict(
        pixel_size=10e-6,
        resolution=(400, 300),
        counts_per_pc=1e4,
        dark_offset=0.0,
        read_noise=0.0,
        bits=16,
    )
    settings.update(overrides)
    return Screen(**settings)


def test_image_counts_track_the_charge_on_the_screen(beam):
    screen = quiet_screen()
    image = screen.measure(beam, rng=0)
    charge_pc = image.counts.sum() / screen.counts_per_pc
    assert charge_pc == pytest.approx(TOTAL_CHARGE * 1e12, rel=1e-3)
    assert image.off_screen_fraction == pytest.approx(0.0, abs=1e-6)


def test_measured_size_is_unbiased_over_many_shots(beam):
    """The single-shot RMS is noisy because the tails are noise-dominated, but
    it must not be *biased*. This is the test that caught rectifying the
    background-subtracted residual at zero, which inflated widths by ~15%."""
    screen = Screen(pixel_size=10e-6, resolution=(400, 300), psf_sigma=25e-6, counts_per_pc=1e4)
    sizes = np.array([screen.measure(beam, rng=seed).beam_size() for seed in range(30)])
    assert sizes[:, 0].mean() == pytest.approx(beam.sigma_x, rel=0.02)
    assert sizes[:, 1].mean() == pytest.approx(beam.sigma_y, rel=0.02)


def test_psf_adds_in_quadrature():
    beam = gaussian_beam(sigma_x=100e-6, sigma_y=100e-6)
    psf = 60e-6
    image = quiet_screen(psf_sigma=psf).measure(beam, rng=0)
    raw_x, _ = image.beam_size(deconvolved=False)
    assert raw_x == pytest.approx(np.hypot(beam.sigma_x, psf), rel=0.02)
    assert image.beam_size()[0] == pytest.approx(beam.sigma_x, rel=0.03)


def test_a_beam_smaller_than_the_psf_reads_as_unresolved():
    """A beam far below the resolution must read as the PSF, and deconvolve
    down to something negligible against it --- never as a negative sqrt."""
    psf = 100e-6
    beam = gaussian_beam(sigma_x=1e-6, sigma_y=1e-6, n=50_000)
    image = quiet_screen(pixel_size=20e-6, psf_sigma=psf).measure(beam, rng=0)
    assert image.beam_size(deconvolved=False)[0] == pytest.approx(psf, rel=0.02)
    assert image.beam_size()[0] < 0.15 * psf


def test_off_screen_charge_is_reported():
    beam = gaussian_beam(centroid_x=5e-3)  # sensor is only +/- 2 mm wide
    image = quiet_screen().measure(beam, rng=0)
    assert image.off_screen_fraction > 0.9


def test_saturation_is_visible():
    image = Screen(
        pixel_size=10e-6, resolution=(200, 200), counts_per_pc=1e6, bits=12
    ).measure(gaussian_beam(sigma_x=50e-6, sigma_y=50e-6), rng=0)
    assert image.saturated_fraction > 0
    assert image.counts.max() == 4095


def test_same_seed_gives_the_same_frame(beam):
    screen = Screen(pixel_size=10e-6, resolution=(100, 100))
    assert np.array_equal(screen.measure(beam, rng=7).counts, screen.measure(beam, rng=7).counts)
    assert not np.array_equal(
        screen.measure(beam, rng=7).counts, screen.measure(beam, rng=8).counts
    )


def test_tilt_compresses_the_horizontal_pixel_scale():
    screen = Screen(pixel_size=10e-6, tilt=np.pi / 4)
    assert screen.beam_pixel_size[0] == pytest.approx(10e-6 / np.sqrt(2))
    assert screen.beam_pixel_size[1] == pytest.approx(10e-6)


def test_screen_geometry_is_validated():
    with pytest.raises(ValueError, match="pixel_size"):
        Screen(pixel_size=0.0)
    with pytest.raises(ValueError, match="resolution"):
        Screen(resolution=(0, 10))


def test_profile_moments_on_an_empty_profile():
    axis = np.linspace(-1, 1, 11)
    assert np.isnan(profile_moments(axis, np.zeros(11))).all()


def test_deconvolve_floors_at_zero():
    assert deconvolve(1.0, 2.0) == 0.0
    assert deconvolve(5.0, 3.0) == pytest.approx(4.0)


# -- Spectrometer ---------------------------------------------------------


def dispersed_beam(dispersion=0.5, relative_spread=1e-3, betatron_size=20e-6):
    """A beam whose horizontal position encodes its momentum, as after a dipole."""
    beam = gaussian_beam(sigma_x=betatron_size, sigma_y=50e-6, relative_energy_spread=0.0)
    rng = np.random.default_rng(42)
    delta = rng.normal(0.0, relative_spread, beam.n)
    beam.x = beam.x + dispersion * delta
    beam.pz = beam.energy * (1.0 + delta)
    return beam


def test_spectrometer_recovers_the_energy_spread():
    dispersion, spread, betatron = 0.5, 1e-3, 20e-6
    beam = dispersed_beam(dispersion, spread, betatron)
    spectrometer = Spectrometer(
        screen=quiet_screen(pixel_size=20e-6, resolution=(400, 200)),
        dispersion=dispersion,
        reference_energy=beam.energy,
    )
    image = spectrometer.measure(beam, rng=0)
    measured = spectrometer.energy_spread(image, beam_size=betatron)
    assert measured == pytest.approx(beam.energy * spread, rel=0.05)
    assert spectrometer.mean_energy(image) == pytest.approx(beam.energy, rel=1e-3)


def test_spectrometer_without_the_betatron_size_reads_high():
    """Not subtracting the beam size gives an upper limit, never an under-read."""
    beam = dispersed_beam(betatron_size=200e-6)
    spectrometer = Spectrometer(
        screen=quiet_screen(pixel_size=20e-6, resolution=(400, 200)),
        dispersion=0.5,
        reference_energy=beam.energy,
    )
    image = spectrometer.measure(beam, rng=0)
    assert spectrometer.energy_spread(image, beam_size=0.0) > spectrometer.energy_spread(
        image, beam_size=200e-6
    )


def test_spectrum_axis_is_monotonic_and_energy_calibrated():
    beam = dispersed_beam()
    spectrometer = Spectrometer(
        screen=quiet_screen(pixel_size=20e-6, resolution=(400, 200)),
        dispersion=-0.5,  # negative dispersion must still come back sorted
        reference_energy=beam.energy,
    )
    energy, intensity = spectrometer.spectrum(spectrometer.measure(beam, rng=0))
    assert np.all(np.diff(energy) > 0)
    assert energy[np.argmax(intensity)] == pytest.approx(beam.energy, rel=1e-2)


def test_spectrometer_rejects_zero_dispersion():
    with pytest.raises(ValueError, match="dispersion"):
        Spectrometer(screen=Screen(), dispersion=0.0, reference_energy=1e8)


# -- StreakedScreen -------------------------------------------------------


def streaked_beam(shear=3e8, unstreaked_size=30e-6):
    """A beam whose vertical position encodes arrival time, as after a TDS."""
    beam = gaussian_beam(sigma_x=50e-6, sigma_y=unstreaked_size, sigma_t=SIGMA_T)
    beam.y = beam.y + shear * beam.t
    return beam


def test_streaked_screen_recovers_the_bunch_length():
    shear, unstreaked = 3e8, 30e-6
    beam = streaked_beam(shear, unstreaked)
    tds = StreakedScreen(
        screen=quiet_screen(pixel_size=20e-6, resolution=(200, 400)), shear=shear
    )
    image = tds.measure(beam, rng=0)
    assert tds.bunch_length(image, unstreaked_size=unstreaked) == pytest.approx(SIGMA_T, rel=0.05)


def test_streak_profile_is_time_calibrated():
    shear = 3e8
    tds = StreakedScreen(
        screen=quiet_screen(pixel_size=20e-6, resolution=(200, 400)), shear=shear
    )
    time, intensity = tds.profile(tds.measure(streaked_beam(shear), rng=0))
    assert np.all(np.diff(time) > 0)
    _, rms = profile_moments(time, intensity)
    assert rms == pytest.approx(SIGMA_T, rel=0.1)


def test_streak_resolution_is_pixel_pitch_over_shear():
    tds = StreakedScreen(screen=Screen(pixel_size=10e-6), shear=3e8)
    assert tds.resolution == pytest.approx(10e-6 / 3e8)


def test_streaked_screen_rejects_zero_shear():
    with pytest.raises(ValueError, match="shear"):
        StreakedScreen(screen=Screen(), shear=0.0)


# -- region of interest ---------------------------------------------------


def test_a_small_beam_on_a_large_sensor_is_still_measurable():
    """The case the region of interest exists for.

    A 46 um beam on a 640x480 sensor covers about four pixels in x, so the
    x-projection is a handful of signal bins among six hundred noise bins.
    Without an iterated region of interest the estimate is worthless --- it
    returned 197 um, then 0.0, then a NaN across neighbouring sensitivities.
    """
    beam = gaussian_beam(sigma_x=46e-6, sigma_y=110e-6)
    screen = Screen(
        pixel_size=12e-6, resolution=(640, 480), psf_sigma=30e-6,
        counts_per_pc=3e3, dark_offset=100.0, read_noise=6.0, bits=12,
    )
    sizes = np.array([screen.measure(beam, rng=seed).beam_size() for seed in range(10)])
    assert np.all(np.isfinite(sizes))
    assert sizes[:, 0].mean() == pytest.approx(beam.sigma_x, rel=0.05)
    assert sizes[:, 1].mean() == pytest.approx(beam.sigma_y, rel=0.05)
    assert sizes[:, 0].std() < 0.05 * beam.sigma_x, "the estimate must also be stable"


def test_disabling_the_region_of_interest_is_noisier_on_a_large_sensor():
    """roi_sigma=None keeps the whole axis: unbiased, but far noisier."""
    beam = gaussian_beam(sigma_x=46e-6, sigma_y=110e-6)
    screen = Screen(
        pixel_size=12e-6, resolution=(640, 480), counts_per_pc=3e3,
        dark_offset=100.0, read_noise=6.0, bits=12,
    )
    images = [screen.measure(beam, rng=seed) for seed in range(10)]
    with_roi = np.array([im.moments_x()[1] for im in images])
    without = np.array([im.moments_x(roi_sigma=None)[1] for im in images])
    assert np.nanstd(without) > 10 * np.nanstd(with_roi)


def test_the_region_of_interest_does_not_bias_a_well_filled_frame(beam):
    """On a frame the beam actually fills, the ROI must change nothing much."""
    screen = Screen(pixel_size=10e-6, resolution=(400, 300), psf_sigma=25e-6, counts_per_pc=1e4)
    images = [screen.measure(beam, rng=seed) for seed in range(20)]
    with_roi = np.mean([im.beam_size()[0] for im in images])
    without = np.mean([im.beam_size(roi_sigma=None)[0] for im in images])
    assert with_roi == pytest.approx(without, rel=0.03)
    assert with_roi == pytest.approx(beam.sigma_x, rel=0.03)


def test_saturation_biases_the_measured_size_wide():
    """A saturated core flattens the profile, so the RMS reads high. This is a
    real measurement error, not a modelling artefact --- check the
    saturated_fraction before trusting a size."""
    beam = gaussian_beam(sigma_x=46e-6, sigma_y=110e-6)
    settings = dict(pixel_size=12e-6, resolution=(640, 480), psf_sigma=30e-6, bits=12)
    clean = Screen(counts_per_pc=3e3, **settings).measure(beam, rng=0)
    hot = Screen(counts_per_pc=3e4, **settings).measure(beam, rng=0)
    assert clean.saturated_fraction == 0.0
    assert hot.saturated_fraction > 0.0
    assert hot.beam_size()[0] > clean.beam_size()[0]
