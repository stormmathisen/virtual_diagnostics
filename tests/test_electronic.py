"""Current monitors, BPMs and the coherent radiation monitor."""

import numpy as np
import pytest
from conftest import SIGMA_T, TOTAL_CHARGE, gaussian_beam

from virtual_diagnostics import CoherentRadiationMonitor, CurrentMonitor, LogAmpReadout


# -- CurrentMonitor -------------------------------------------------------


def test_a_monitor_without_droop_integrates_to_the_true_charge(beam):
    fct = CurrentMonitor(
        rise_time=200e-12, droop_time=None, transimpedance=1.0, noise=0.0,
        sample_rate=2e12, duration=20e-9,
    )
    t, v = fct.measure(beam, rng=0)
    assert fct.integrated_charge(t, v) == pytest.approx(TOTAL_CHARGE, rel=1e-3)


def test_a_fast_monitor_follows_the_instantaneous_current(beam):
    """Rise time well below the bunch length: the peak is the peak beam current,
    Q / (sqrt(2 pi) sigma_t) for a Gaussian."""
    fct = CurrentMonitor(
        rise_time=100e-15, droop_time=None, transimpedance=1.0, noise=0.0,
        sample_rate=2e13, duration=20e-12,
    )
    _, v = fct.measure(beam, rng=0)
    assert v.max() == pytest.approx(TOTAL_CHARGE / (np.sqrt(2 * np.pi) * SIGMA_T), rel=0.05)


def test_an_integrating_monitor_turns_charge_into_pulse_height(beam):
    """An ICT's rise time is far longer than the bunch, so the output is the
    impulse response scaled by charge: peak = Q / tau, area = Q."""
    ict = CurrentMonitor(
        rise_time=20e-9, droop_time=None, transimpedance=1.0, noise=0.0,
        sample_rate=2e9, duration=2e-6,
    )
    t, v = ict.measure(beam, rng=0)
    tau = ict.rise_time / np.log(9.0)
    assert v.max() == pytest.approx(TOTAL_CHARGE / tau, rel=0.1)
    assert ict.integrated_charge(t, v) == pytest.approx(TOTAL_CHARGE, rel=1e-2)

    doubled = ict.measure(beam.scaled_charge(2 * TOTAL_CHARGE), rng=0)[1]
    assert doubled.max() == pytest.approx(2 * v.max(), rel=1e-6)


def test_droop_makes_an_integrating_monitor_read_low(beam):
    """The high pass removes DC, so the pulse is followed by an undershoot of
    equal area. Integrating over a finite record therefore under-reads, and it
    under-reads more the shorter the droop time. This is the real error the
    hardware droop correction exists to fix."""
    def integrated(droop):
        ict = CurrentMonitor(
            rise_time=20e-9, droop_time=droop, transimpedance=1.0, noise=0.0,
            sample_rate=2e9, duration=1e-6,
        )
        t, v = ict.measure(beam, rng=0)
        return ict.integrated_charge(t, v)

    readings = [integrated(d) for d in (1e-6, 5e-6, 50e-6)]
    assert readings[0] < readings[1] < readings[2] < TOTAL_CHARGE
    assert readings[2] == pytest.approx(TOTAL_CHARGE, rel=0.05)


def test_bandwidth_follows_from_the_rise_time():
    monitor = CurrentMonitor(rise_time=1e-9)
    assert monitor.bandwidth == pytest.approx(np.log(9.0) / (2 * np.pi * 1e-9))


def test_digitised_output_is_integer_codes(beam):
    monitor = CurrentMonitor(bits=12, full_scale=0.05, noise=0.0)
    _, codes = monitor.measure(beam, rng=0)
    assert codes.dtype.kind == "u"
    assert codes.max() <= 4095


def test_monitor_settings_are_validated():
    with pytest.raises(ValueError, match="rise_time"):
        CurrentMonitor(rise_time=0.0)
    with pytest.raises(ValueError, match="pretrigger"):
        CurrentMonitor(pretrigger=1.0)
    with pytest.raises(ValueError, match="droop_time"):
        CurrentMonitor(droop_time=-1e-6).response(np.zeros(10))


# -- log amp --------------------------------------------------------------


def test_log_amp_round_trips():
    amp = LogAmpReadout(qcal=1e-12, ucal=0.5)
    for charge in (1e-12, 50e-12, 250e-12, 1e-9):
        assert amp.charge_from_voltage(amp.voltage_from_charge(charge)) == pytest.approx(charge)


def test_log_amp_is_linear_in_volts_per_decade():
    amp = LogAmpReadout(qcal=1e-12, ucal=0.5)
    step = amp.voltage_from_charge(1e-9) - amp.voltage_from_charge(100e-12)
    assert step == pytest.approx(0.5)


def test_log_amp_has_no_answer_for_no_beam():
    amp = LogAmpReadout(qcal=1e-12, ucal=0.5)
    assert amp.voltage_from_charge(0.0) == -np.inf
    assert np.isnan(amp.voltage_from_charge(-1e-12))


def test_measure_charge_needs_a_readout(beam):
    with pytest.raises(ValueError, match="LogAmpReadout"):
        CurrentMonitor().measure_charge(beam)


def test_measure_charge_matches_the_integrated_waveform(beam):
    ict = CurrentMonitor(
        rise_time=20e-9, droop_time=None, transimpedance=1.0, noise=0.0,
        sample_rate=2e9, duration=2e-6, readout=LogAmpReadout(qcal=1e-12, ucal=0.5),
    )
    assert ict.measure_charge(beam, rng=0) == pytest.approx(TOTAL_CHARGE, rel=1e-2)


# -- CoherentRadiationMonitor ---------------------------------------------


def test_form_factor_is_one_at_zero_frequency(beam):
    frequency, magnitude = CoherentRadiationMonitor().form_factor(beam)
    assert frequency[0] == 0.0
    assert magnitude[0] == pytest.approx(1.0, rel=1e-6)


def test_form_factor_matches_the_gaussian_analytic_result():
    """For a Gaussian bunch, |F(f)|**2 = exp(-(2 pi f sigma_t)**2)."""
    beam = gaussian_beam(n=400_000, sigma_t=300e-15)
    frequency, magnitude = CoherentRadiationMonitor(band=(0.1e12, 2e12)).form_factor(beam)
    probe = (frequency > 0.2e12) & (frequency < 1.0e12)
    analytic = np.exp(-((2 * np.pi * frequency[probe] * beam.sigma_t) ** 2))
    assert magnitude[probe] == pytest.approx(analytic, rel=0.1)


def test_signal_rises_steeply_as_the_bunch_shortens():
    monitor = CoherentRadiationMonitor(band=(0.3e12, 3e12), calibration=1e18)
    signals = [
        monitor.measure(gaussian_beam(n=200_000, sigma_t=sigma), rng=0)
        for sigma in (1000e-15, 500e-15, 200e-15)
    ]
    assert signals[0] < signals[1] < signals[2]
    assert signals[2] / signals[0] > 100


def test_signal_scales_with_charge_squared():
    monitor = CoherentRadiationMonitor(calibration=1e18)
    beam = gaussian_beam(n=200_000, sigma_t=200e-15)
    single = monitor.measure(beam, rng=0)
    doubled = monitor.measure(beam.scaled_charge(2 * TOTAL_CHARGE), rng=0)
    assert doubled == pytest.approx(4 * single, rel=1e-6)


def test_band_is_validated():
    with pytest.raises(ValueError, match="band"):
        CoherentRadiationMonitor(band=(3e12, 1e12))
