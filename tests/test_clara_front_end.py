"""The CLARA analog front end example (IBIC2022 MOP32).

Locks in the two amplitudes the paper actually measured, so a change to the
netlist or the monitor models cannot quietly drift away from them.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from virtual_diagnostics import SpiceFrontEnd, ngspice_available

pytestmark = pytest.mark.skipif(
    not ngspice_available(), reason="needs the ngspice binary (apt install ngspice)"
)

ROOT = Path(__file__).resolve().parent.parent
NETLIST = ROOT / "examples" / "netlists" / "clara_front_end.cir"


@pytest.fixture(scope="module")
def example():
    """The example script, imported as a module."""
    spec = importlib.util.spec_from_file_location(
        "clara_front_end", ROOT / "examples" / "clara_front_end.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deliver(charge, cf, rf, gout, sigma=2e-9):
    """Push a known charge through the front end, return the output waveform."""
    netlist = NETLIST.read_text()
    netlist = netlist.replace(".param CF=1n", f".param CF={cf:.6e}")
    netlist = netlist.replace(".param RF=1k", f".param RF={rf:.6e}")
    netlist = netlist.replace(".param GOUT=2.8", f".param GOUT={gout:.6e}")

    dt, n = 2e-10, 20_000
    t = np.arange(n) * dt
    current = charge / (np.sqrt(2 * np.pi) * sigma) * np.exp(
        -((t - 200e-9) ** 2) / (2 * sigma**2)
    )
    front_end = SpiceFrontEnd(netlist=netlist, sources="Vin", outputs="out",
                              max_points=20_000)
    _, v = front_end.run(t, 50.0 * current)
    return v


def peak(v):
    return float(v[np.abs(v).argmax()])


def test_the_integrator_returns_minus_gain_times_charge_over_capacitance():
    """The operating principle: peak output is -GOUT * Q / CF, and the paper's
    claim that bunch charge is proportional to the peak depends on it."""
    v = deliver(100e-12, cf=1e-9, rf=1e3, gout=2.8)
    assert peak(v) == pytest.approx(-2.8 * 100e-12 / 1e-9, rel=0.03)


def test_the_peak_does_not_depend_on_the_pulse_shape():
    """Which is the whole point of integrating: a charge measurement must not
    care how long the bunch took to arrive."""
    peaks = [peak(deliver(100e-12, 1e-9, 1e3, 2.8, sigma=s)) for s in (1e-9, 4e-9)]
    assert peaks[0] == pytest.approx(peaks[1], rel=0.05)


def test_output_is_linear_in_charge():
    charges = np.array([25e-12, 100e-12, 250e-12])
    peaks = np.array([abs(peak(deliver(q, 1e-9, 1e3, 2.8))) for q in charges])
    slope, intercept = np.polyfit(charges, peaks, 1)
    residual = peaks - (slope * charges + intercept)
    assert np.abs(residual).max() < 0.01 * peaks.max()


def test_every_table_1_setting_reaches_full_scale_at_the_top_of_its_range(example):
    """Table 1 pairs each capacitance with an operating range; the fitted gains
    should put the top of each range near the full scale seen in Fig. 3."""
    for name, (cf, rf, gout, (_, high)) in example.SETTINGS.items():
        v = abs(peak(deliver(high * 1e-12, cf, rf, gout)))
        assert 0.4 * example.FULL_SCALE < v < 1.2 * example.FULL_SCALE, name


def test_a_faraday_cup_reproduces_the_measured_amplitude(example):
    """Fig. 4: about -300 mV for 100 pC on the lowest sensitivity setting.

    Nothing is fitted here -- a cup discharges its whole charge through 50 ohm,
    so this amplitude falls out of Table 1's capacitance and the fitted gain.
    """
    front_end = example.front_end("lowest")
    t, v_in = example.faraday_cup.measure(example.bunch(100e-12), rng=example.SEED)
    _, v_out = front_end.run(t, v_in)

    assert peak(v_out) < 0, "a Faraday cup gives a negative output (Fig. 4)"
    assert peak(v_out) == pytest.approx(-0.300, abs=0.05)


def test_a_wall_current_monitor_flips_the_polarity(example):
    """Fig. 5: about +400 mV, positive, because a WCM sees the image current --
    and taken on the highest setting because its signal is so much smaller."""
    front_end = example.front_end("highest")
    t, v_in = example.wall_current_monitor.measure(example.bunch(100e-12), rng=example.SEED)
    _, v_out = front_end.run(t, v_in)

    assert peak(v_out) > 0, "a WCM gives a positive output (Fig. 5)"
    # An approximate reproduction: the WCM's transfer impedance is not given in
    # the paper and is set to a round 2 ohm rather than tuned to match, which
    # lands about 13 % below the reported amplitude.
    assert 0.30 < peak(v_out) < 0.45


def test_the_wcm_delivers_only_a_fraction_of_the_charge(example):
    """Zt/50 of it, which is why the paper uses the highest sensitivity."""
    t, v_in = example.wall_current_monitor.measure(example.bunch(100e-12), rng=example.SEED)
    delivered = np.trapezoid(v_in, t) / 50.0
    assert abs(delivered) == pytest.approx(
        100e-12 * example.WCM_TRANSFER_IMPEDANCE / 50.0, rel=0.05
    )


def test_the_low_frequency_corner_is_recoverable_from_the_waveform(example):
    """A high pass removes DC: the bunch is followed by an opposite-sign tail of
    equal area decaying with 1/(2 pi f_low)."""
    from virtual_diagnostics import CurrentMonitor

    wcm = example.wall_current_monitor
    slow = CurrentMonitor(
        rise_time=wcm.rise_time, droop_time=wcm.droop_time,
        transimpedance=wcm.transimpedance, noise=0.0,
        sample_rate=2e9, duration=8e-6, pretrigger=0.05,
    )
    t, v = slow.measure(example.bunch(100e-12), rng=example.SEED)
    at = int(v.argmin())

    pulse_area = np.trapezoid(v[: at + 2], t[: at + 2])
    tail_area = np.trapezoid(v[at + 2 :], t[at + 2 :])
    assert abs(pulse_area + tail_area) < 0.02 * abs(pulse_area), "areas must cancel"

    tail_t, tail_v = t[at + 20 :], v[at + 20 :]
    usable = tail_v > tail_v.max() * 0.05
    fitted = -1.0 / np.polyfit(tail_t[usable], np.log(tail_v[usable]), 1)[0]
    assert fitted == pytest.approx(1.0 / (2 * np.pi * example.WCM_LOW), rel=0.01)
