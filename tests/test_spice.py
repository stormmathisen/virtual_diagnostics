"""The SPICE front end. Skipped unless ngspice is actually usable."""

import numpy as np
import pytest

from virtual_diagnostics import SpiceFrontEnd, ngspice_available

pytestmark = pytest.mark.skipif(
    not ngspice_available(), reason="needs PySpice and the ngspice shared library"
)

RC_LOW_PASS = """* 50 ohm source into an RC low pass
Vin inp 0 PWL(0 0)
R1 inp out 50
C1 out 0 20p
Rload out 0 1e6
.tran 1p 1n
.end
"""

SUMMING = """* two sources, one summed output
Va a 0 PWL(0 0)
Vb b 0 PWL(0 0)
Ra a sum 50
Rb b sum 50
Rterm sum 0 50
Eout out 0 sum 0 1.0
Rload out 0 1e6
.end
"""

DELAY_LINE = """* two electrodes joined by a delay line, tapped at the near end
Va a 0 PWL(0 0)
Vb b 0 PWL(0 0)
Ra a near 50
Rb b far 50
Tline near 0 far 0 Z0=50 TD=2n
Rout near 0 50
Eout out 0 near 0 1.0
Rload out 0 1e6
.end
"""


def pulse(width=2e-10, start=-1e-9, stop=8e-9, points=801, amplitude=1.0):
    t = np.linspace(start, stop, points)
    return t, amplitude * np.exp(-((t / width) ** 2))


# -- single source --------------------------------------------------------


def test_a_low_pass_attenuates_and_delays():
    t, v = pulse()
    front_end = SpiceFrontEnd(netlist=RC_LOW_PASS, sources="Vin", outputs="out")
    t_out, v_out = front_end.run(t, v)

    assert t_out == pytest.approx(t), "the result comes back on the input time axis"
    assert v_out.max() < v.max(), "an RC low pass attenuates a pulse this short"
    assert t_out[v_out.argmax()] > t[v.argmax()], "and delays it"


def test_negative_input_times_survive_the_round_trip():
    """SPICE starts at t=0 and discards anything before it. Beam time axes run
    negative, so the record is shifted in and shifted back out."""
    t, v = pulse(start=-1e-9)
    assert t[0] < 0
    t_out, v_out = SpiceFrontEnd(netlist=RC_LOW_PASS).run(t, v)
    assert t_out[0] == pytest.approx(t[0])
    assert v_out[t_out < -0.8e-9].max() < 0.05, "the quiet part stays quiet"


def test_gain_is_applied():
    netlist = """* x10 amplifier
Vin inp 0 PWL(0 0)
Eout out 0 inp 0 10.0
Rload out 0 1e6
.end
"""
    t, v = pulse()
    _, v_out = SpiceFrontEnd(netlist=netlist).run(t, v)
    assert v_out.max() == pytest.approx(10 * v.max(), rel=0.02)


# -- several sources in one simulation ------------------------------------


def test_two_sources_are_driven_in_the_same_run():
    """The reason sources are plural: the netlist has to see them together
    before it can combine them."""
    t, v = pulse()
    front_end = SpiceFrontEnd(netlist=SUMMING, sources=("Va", "Vb"), outputs="out")

    _, both = front_end.run(t, np.stack([v, v]))
    _, only_a = front_end.run(t, np.stack([v, np.zeros_like(v)]))
    assert both.max() == pytest.approx(2 * only_a.max(), rel=0.02)


def test_signals_can_be_given_by_name():
    t, v = pulse()
    front_end = SpiceFrontEnd(netlist=SUMMING, sources=("Va", "Vb"), outputs="out")
    _, by_array = front_end.run(t, np.stack([v, 0.5 * v]))
    _, by_name = front_end.run(t, {"Va": v, "Vb": 0.5 * v})
    assert by_name == pytest.approx(by_array)


def test_a_delay_line_in_the_netlist_produces_two_peaks():
    """The topology this exists for: the near electrode arrives directly, the
    far one arrives a line delay later, on one trace."""
    t, near = pulse(amplitude=1.0)
    _, far = pulse(amplitude=0.6)
    front_end = SpiceFrontEnd(netlist=DELAY_LINE, sources=("Va", "Vb"), outputs="out")
    t_out, v = front_end.run(t, np.stack([near, far]))

    first = t_out[np.flatnonzero(v >= 0.2 * v.max())[0]]
    window = 0.5e-9
    direct = v[np.abs(t_out - first) <= window].max()
    delayed = v[np.abs(t_out - (first + 2e-9)) <= window].max()

    assert direct > 0 and delayed > 0, "both pulses are on the trace"
    assert direct > delayed, "the larger electrode is the one read directly"
    assert delayed / direct == pytest.approx(0.6, rel=0.25)


def test_the_peak_difference_tracks_the_electrode_imbalance():
    t, base = pulse()
    front_end = SpiceFrontEnd(netlist=DELAY_LINE, sources=("Va", "Vb"), outputs="out")
    ratios = []
    for imbalance in (0.6, 0.8, 1.0, 1.2, 1.4):
        t_out, v = front_end.run(t, np.stack([base, imbalance * base]))
        first = t_out[np.flatnonzero(v >= 0.2 * v.max())[0]]
        window = 0.5e-9
        direct = v[np.abs(t_out - first) <= window].max()
        delayed = v[np.abs(t_out - (first + 2e-9)) <= window].max()
        ratios.append((direct - delayed) / (direct + delayed))
    assert np.all(np.diff(ratios) < 0), "more signal on the far electrode, lower ratio"


def test_several_outputs_come_back_as_a_dict():
    t, v = pulse()
    front_end = SpiceFrontEnd(netlist=SUMMING, sources=("Va", "Vb"), outputs=("sum", "out"))
    _, results = front_end.run(t, np.stack([v, v]))
    assert set(results) == {"sum", "out"}
    assert results["out"] == pytest.approx(results["sum"], abs=1e-9)


def test_button_signals_go_straight_in():
    """ElectrodeSignals.volts is already shaped (n_electrodes, n_samples)."""
    from conftest import gaussian_beam

    from virtual_diagnostics import ButtonBPM

    bpm = ButtonBPM(pipe_radius=20e-3)
    signals = bpm.measure(gaussian_beam(n=20_000, sigma_x=150e-6, sigma_y=150e-6), rng=0)
    front_end = SpiceFrontEnd(
        netlist=SUMMING, sources=("Va", "Vb"), outputs="out"
    )
    _, out = front_end.run(signals.t, signals.volts[:2])
    assert out.shape == signals.t.shape
    assert np.isfinite(out).all()


# -- netlist handling and errors ------------------------------------------


def test_the_built_netlist_is_inspectable_and_decimated():
    t, v = pulse(points=20_000)
    front_end = SpiceFrontEnd(netlist=RC_LOW_PASS, max_points=500)
    built = front_end.build(t - t[0], v)

    assert built.count(".tran") == 1, "the original .tran is replaced, not duplicated"
    assert built.strip().endswith(".end")
    pwl = built[built.index("PWL(") + 4 : built.index(")", built.index("PWL("))]
    assert len(pwl.split()) // 2 <= 500


def test_a_missing_source_is_a_clear_error():
    t, v = pulse()
    with pytest.raises(ValueError, match="no source named"):
        SpiceFrontEnd(netlist=RC_LOW_PASS, sources="Vbeam").run(t, v)


def test_a_missing_output_node_is_a_clear_error():
    t, v = pulse()
    with pytest.raises(RuntimeError, match="not in the results"):
        SpiceFrontEnd(netlist=RC_LOW_PASS, outputs="nowhere").run(t, v)


def test_wrong_number_of_waveforms_is_a_clear_error():
    t, v = pulse()
    front_end = SpiceFrontEnd(netlist=SUMMING, sources=("Va", "Vb"), outputs="out")
    with pytest.raises(ValueError, match="drives 2 sources"):
        front_end.run(t, v)
    with pytest.raises(ValueError, match="unknown sources"):
        front_end.run(t, {"Va": v, "Vc": v})
    with pytest.raises(ValueError, match="no signal given"):
        front_end.run(t, {"Va": v})


def test_input_validation():
    t, v = pulse()
    front_end = SpiceFrontEnd(netlist=RC_LOW_PASS)
    with pytest.raises(ValueError, match="expected"):
        front_end.run(t, v[:-1])
    with pytest.raises(ValueError, match="increasing"):
        front_end.run(t[::-1], v)
    with pytest.raises(ValueError, match="at least two samples"):
        front_end.run(t[:1], v[:1])


def test_a_netlist_can_be_loaded_from_a_file(tmp_path):
    path = tmp_path / "frontend.cir"
    path.write_text(RC_LOW_PASS)
    t, v = pulse()
    _, v_out = SpiceFrontEnd(netlist=str(path)).run(t, v)
    assert np.isfinite(v_out).all()


def test_the_shipped_delay_line_combiner_netlist_runs():
    from conftest import gaussian_beam

    from virtual_diagnostics import ButtonBPM

    bpm = ButtonBPM(pipe_radius=20e-3)
    combiner = SpiceFrontEnd(
        netlist="examples/netlists/delay_line_combiner.cir",
        sources=("Vb1", "Vb2", "Vb3", "Vb4"),
        outputs="xout",
    )
    ratios = []
    for x0 in (-1e-3, 0.0, 1e-3):
        beam = gaussian_beam(n=50_000, sigma_x=150e-6, sigma_y=150e-6, centroid_x=x0)
        signals = bpm.measure(beam, rng=1)
        t, v = combiner.run(signals.t, signals.volts)
        first = t[np.flatnonzero(v >= 0.2 * v.max())[0]]
        window = 0.5e-9
        direct = v[np.abs(t - first) <= window].max()
        delayed = v[np.abs(t - (first + 2e-9)) <= window].max()
        ratios.append((direct - delayed) / (direct + delayed))
    assert np.all(np.diff(ratios) > 0), "the peak difference tracks position"
