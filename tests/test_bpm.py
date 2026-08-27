"""BPMs: the readout model, the per-button model, and the delay line."""

import numpy as np
import pytest
from conftest import TOTAL_CHARGE, gaussian_beam

from virtual_diagnostics import (
    BPM,
    ButtonBPM,
    StriplineBPM,
    wall_current_fraction,
)
from virtual_diagnostics.bpm import DIAGONAL_ELECTRODES


def offset_beam(x0=0.0, y0=0.0, charge=TOTAL_CHARGE, n=100_000):
    return gaussian_beam(
        n=n, sigma_x=150e-6, sigma_y=150e-6, centroid_x=x0, centroid_y=y0, total_charge=charge
    )


# -- readout-level BPM ----------------------------------------------------


def test_bpm_is_unbiased_and_matches_its_own_noise_model():
    bpm = BPM(resolution=10e-6, reference_charge=100e-12)
    beam = offset_beam(200e-6, -50e-6, n=20_000)
    readings = [bpm.measure(beam, rng=seed) for seed in range(300)]
    x = np.array([r.x for r in readings])
    y = np.array([r.y for r in readings])
    expected = bpm.noise_at(TOTAL_CHARGE)

    assert x.mean() == pytest.approx(beam.centroid_x, abs=0.5 * expected)
    assert y.mean() == pytest.approx(beam.centroid_y, abs=0.5 * expected)
    assert x.std() == pytest.approx(expected, rel=0.15)


def test_bpm_resolution_degrades_as_one_over_charge():
    bpm = BPM(resolution=10e-6, reference_charge=100e-12)
    assert bpm.noise_at(100e-12) == pytest.approx(10e-6)
    assert bpm.noise_at(10e-12) == pytest.approx(100e-6)
    assert bpm.noise_at(-100e-12) == pytest.approx(10e-6), "sign of the charge is irrelevant"
    assert bpm.noise_at(0.0) == np.inf


def test_bpm_reports_nan_below_its_charge_threshold():
    bpm = BPM(charge_threshold=1e-12)
    reading = bpm.measure(offset_beam(charge=1e-15, n=1000), rng=0)
    assert np.isnan(reading.x) and np.isnan(reading.y)
    assert reading.charge == pytest.approx(1e-15)


def test_bpm_gain_and_offset_are_calibration_knobs():
    beam = offset_beam(1e-3, n=20_000)
    bpm = BPM(resolution=0.0, gain=(1.1, 1.0), offset=(50e-6, 0.0))
    assert bpm.measure(beam, rng=0).x == pytest.approx(1.1 * beam.centroid_x + 50e-6)


# -- wall current geometry ------------------------------------------------


def test_a_centred_beam_shares_the_current_equally():
    fractions = wall_current_fraction(0.0, 0.0, 20e-3, DIAGONAL_ELECTRODES, 0.5)
    assert fractions == pytest.approx(0.5 / (2 * np.pi))


def test_the_whole_circumference_intercepts_all_of_it():
    """The Poisson kernel integrates to exactly one over 2 pi, at any offset."""
    for x, y in ((0.0, 0.0), (3e-3, 1e-3), (0.0, 15e-3)):
        total = wall_current_fraction(x, y, 20e-3, [0.0], 2 * np.pi).sum()
        assert total == pytest.approx(1.0, rel=1e-6)


def test_the_near_button_sees_more_current():
    fractions = wall_current_fraction(5e-3, 0.0, 20e-3, DIAGONAL_ELECTRODES, 0.5)
    right = fractions[0] + fractions[3]   # buttons at +/- 45 degrees
    left = fractions[1] + fractions[2]
    assert right > left
    assert fractions[0] == pytest.approx(fractions[3]), "symmetric about the x axis"


def test_a_beam_outside_the_pipe_is_rejected():
    with pytest.raises(ValueError, match="outside the pipe"):
        wall_current_fraction(25e-3, 0.0, 20e-3, DIAGONAL_ELECTRODES, 0.5)


# -- per-button BPM -------------------------------------------------------


def calibrated_button_bpm(**overrides):
    bpm = ButtonBPM(pipe_radius=20e-3, **overrides)
    bpm.sensitivity = bpm.calibrate_sensitivity()
    return bpm


def test_button_signals_are_bipolar():
    """A button is capacitively coupled, so its pulse is a spike followed by an
    undershoot. A unipolar button signal means the R*C droop is wrong."""
    signals = calibrated_button_bpm().measure(offset_beam(), rng=0)
    assert signals.volts.shape == (4, signals.t.size)
    assert np.all(signals.volts.max(axis=1) > 0)
    assert np.all(signals.volts.min(axis=1) < 0)


def test_button_bpm_reads_the_true_position_near_the_centre():
    bpm = calibrated_button_bpm()
    for x0 in (0.0, 5e-4, 1e-3):
        x, y = bpm.position(bpm.measure(offset_beam(x0), rng=1))
        assert x == pytest.approx(x0, abs=20e-6)
        assert y == pytest.approx(0.0, abs=20e-6)


def test_button_bpm_goes_nonlinear_towards_the_wall():
    """The point of modelling buttons individually. The Poisson kernel makes the
    projection read progressively low as the beam approaches the pipe wall."""
    bpm = calibrated_button_bpm()
    errors = []
    for x0 in (1e-3, 6e-3, 12e-3):
        x, _ = bpm.position(bpm.measure(offset_beam(x0), rng=1))
        errors.append(abs(x - x0))
    assert errors[0] < 50e-6, "linear near the axis"
    assert errors[0] < errors[1] < errors[2], "and progressively worse towards the wall"
    assert errors[2] > 1e-3


def test_position_is_insensitive_to_electrode_undersampling():
    """An under-sampled one-pole response reports amplitudes that are too small,
    but the error is common to every button and cancels in the ratio."""
    from virtual_diagnostics import CurrentMonitor

    beam = offset_beam(2e-3)
    positions = []
    for sample_rate in (5e10, 2e11, 1e12):
        electrode = CurrentMonitor(
            rise_time=30e-12, droop_time=250e-12, transimpedance=50.0,
            noise=0.0, sample_rate=sample_rate, duration=4e-9, pretrigger=0.15,
        )
        bpm = calibrated_button_bpm(electrode=electrode)
        positions.append(bpm.position(bpm.measure(beam, rng=1))[0])
    assert positions == pytest.approx([positions[0]] * len(positions), rel=1e-6)


def test_button_gain_mismatch_shifts_the_reading():
    beam = offset_beam(0.0)
    perfect = calibrated_button_bpm()
    mismatched = calibrated_button_bpm(gains=(1.05, 1.0, 1.0, 1.0))
    assert perfect.position(perfect.measure(beam, rng=1))[0] == pytest.approx(0.0, abs=20e-6)
    assert abs(mismatched.position(mismatched.measure(beam, rng=1))[0]) > 100e-6


def test_integral_readout_agrees_with_peak_readout():
    bpm = calibrated_button_bpm()
    signals = bpm.measure(offset_beam(2e-3), rng=1)
    by_peak = bpm.position(signals, readout="peak")[0]
    by_integral = bpm.position(signals, readout="integral")[0]
    assert by_integral == pytest.approx(by_peak, rel=0.02)
    with pytest.raises(ValueError, match="readout"):
        bpm.amplitudes(signals, readout="rms")


def test_button_bpm_geometry_is_validated():
    with pytest.raises(ValueError, match="pipe_radius"):
        ButtonBPM(pipe_radius=0.0)
    with pytest.raises(ValueError, match="electrode_width"):
        ButtonBPM(electrode_width=0.0)
    with pytest.raises(ValueError, match="at least two electrodes"):
        ButtonBPM(electrode_angles=(0.0,))
    with pytest.raises(ValueError, match="gains"):
        ButtonBPM(gains=(1.0, 1.0))


# -- striplines -----------------------------------------------------------


def calibrated_stripline(length=100e-3, **overrides):
    bpm = StriplineBPM(pipe_radius=20e-3, length=length, **overrides)
    bpm.sensitivity = bpm.calibrate_sensitivity()
    return bpm


def test_stripline_pulses_are_a_round_trip_apart_and_opposite():
    """The defining behaviour: a pulse in as the bunch enters, an equal and
    opposite one as it leaves, separated by L/(beta c) + L/c."""
    bpm = calibrated_stripline()
    beam = offset_beam()
    signals = bpm.measure(beam, rng=0)
    v = signals.volts[0]

    tau = bpm.round_trip_time(beam.relativistic_beta)
    assert tau == pytest.approx(2 * 100e-3 / 299792458.0, rel=1e-3), "2L/c for a fast beam"

    separation = signals.t[int(np.argmin(v))] - signals.t[int(np.argmax(v))]
    assert separation == pytest.approx(tau, rel=0.01)
    assert v.max() == pytest.approx(-v.min(), rel=0.02), "equal and opposite"


def test_a_stripline_has_no_dc_response():
    """The two pulses have equal area whatever the bunch shape, so the waveform
    integrates to zero. A stripline simply cannot measure charge."""
    bpm = calibrated_stripline()
    for sigma_t in (1e-12, 5e-12):
        beam = gaussian_beam(n=100_000, sigma_x=150e-6, sigma_y=150e-6, sigma_t=sigma_t)
        signals = bpm.measure(beam, rng=0)
        area = np.trapezoid(signals.volts[0], signals.t)
        scale = np.abs(signals.volts[0]).max() * bpm.round_trip_time()
        assert abs(area) < 1e-6 * scale


def test_stripline_comb_response():
    """|H(f)| = 2 |sin(pi f tau)|: peak at the quarter-wave frequency c/(4L),
    nulls at every multiple of 1/tau. Cutting the length is how you tune it."""
    bpm = calibrated_stripline()
    f0 = bpm.quarter_wave_frequency
    assert f0 == pytest.approx(299792458.0 / (4 * 100e-3), rel=1e-3)

    assert bpm.transfer_magnitude(f0) == pytest.approx(2.0)
    assert bpm.transfer_magnitude(0.0) == pytest.approx(0.0, abs=1e-12)
    assert bpm.transfer_magnitude(2 * f0) == pytest.approx(0.0, abs=1e-9), "first null"
    assert bpm.transfer_magnitude(3 * f0) == pytest.approx(2.0), "next lobe"


def test_a_shorter_stripline_peaks_higher_in_frequency():
    assert calibrated_stripline(length=50e-3).quarter_wave_frequency == pytest.approx(
        2 * calibrated_stripline(length=100e-3).quarter_wave_frequency
    )


def test_stripline_reads_position_and_goes_nonlinear_like_a_button():
    """The geometry is shared, so the position behaviour must be too."""
    bpm = calibrated_stripline()
    assert bpm.position(bpm.measure(offset_beam(1e-3), rng=1))[0] == pytest.approx(
        1e-3, abs=20e-6
    )
    far = bpm.position(bpm.measure(offset_beam(12e-3), rng=1))[0]
    assert abs(far - 12e-3) > 1e-3


def test_stripline_directivity_suppresses_the_far_port():
    bpm = calibrated_stripline(directivity=26.0)
    beam = offset_beam()
    upstream = bpm.measure(beam, rng=1).volts[0].max()
    downstream = bpm.measure(beam, rng=1, port="downstream").volts[0].max()
    assert upstream / downstream == pytest.approx(10 ** (26 / 20), rel=0.02)
    with pytest.raises(ValueError, match="port"):
        bpm.measure(beam, rng=1, port="sideways")


def test_a_record_shorter_than_the_round_trip_is_rejected():
    """Otherwise the second pulse falls off the end and a stripline silently
    looks like a button."""
    from virtual_diagnostics import CurrentMonitor

    short = CurrentMonitor(
        rise_time=20e-12, droop_time=None, transimpedance=50.0,
        noise=0.0, sample_rate=1e12, duration=200e-12,
    )
    bpm = StriplineBPM(pipe_radius=20e-3, length=100e-3, electrode=short)
    with pytest.raises(ValueError, match="round trip"):
        bpm.measure(offset_beam(), rng=0)


def test_stripline_geometry_is_validated():
    with pytest.raises(ValueError, match="length"):
        StriplineBPM(length=0.0)
    with pytest.raises(ValueError, match="impedance"):
        StriplineBPM(impedance=0.0)
