"""The SPICE front end.

Split deliberately: the rawfile parser, netlist construction and the
not-installed path are tested against committed fixtures and need nothing
installed. Only the end-to-end tests require the ngspice binary.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from virtual_diagnostics import (
    NgspiceError,
    NgspiceNotFound,
    SpiceFrontEnd,
    ngspice_available,
    ngspice_executable,
    ngspice_version,
    read_rawfile,
)

DATA = Path(__file__).parent / "data"

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


# -- rawfile parsing (fixtures, no ngspice needed) ------------------------


def test_binary_and_ascii_rawfiles_parse_identically():
    """Both fixtures are real ngspice 42 output for the same RC circuit."""
    binary = read_rawfile(DATA / "transient_binary.raw")
    ascii_ = read_rawfile(DATA / "transient_ascii.raw")

    assert set(binary) == {"time", "v(inp)", "v(out)", "i(vin)"}
    assert set(binary) == set(ascii_)
    for name in binary:
        assert binary[name] == pytest.approx(ascii_[name])


def test_rawfile_values_are_physical():
    """A 1 ns RC driven by a 1 V triangle peaking at 1 ns: the output lags and
    is attenuated, and never exceeds the input."""
    results = read_rawfile(DATA / "transient_binary.raw")
    time, inp, out = results["time"], results["v(inp)"], results["v(out)"]

    assert time.size == 66
    assert np.all(np.diff(time) > 0)
    assert out.max() < inp.max()
    assert time[out.argmax()] > time[inp.argmax()]
    assert out.max() == pytest.approx(0.509962, rel=1e-4)


def test_a_file_that_is_not_a_rawfile_is_rejected(tmp_path):
    bad = tmp_path / "nonsense.raw"
    bad.write_bytes(b"this is not a rawfile")
    with pytest.raises(NgspiceError, match="not an ngspice rawfile"):
        read_rawfile(bad)


def test_a_truncated_rawfile_is_rejected(tmp_path):
    raw = (DATA / "transient_binary.raw").read_bytes()
    cut = tmp_path / "cut.raw"
    cut.write_bytes(raw[: len(raw) - 400])
    with pytest.raises(NgspiceError, match="truncated"):
        read_rawfile(cut)


# -- locating ngspice, and failing well without it ------------------------


def test_the_executable_can_be_pointed_at_explicitly(tmp_path, monkeypatch):
    fake = tmp_path / "ngspice"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("NGSPICE_EXECUTABLE", str(fake))
    assert ngspice_executable() == str(fake)
    assert ngspice_available()


def test_a_bogus_explicit_path_is_not_accepted(monkeypatch):
    monkeypatch.setenv("NGSPICE_EXECUTABLE", "/nowhere/ngspice")
    assert ngspice_executable() is None
    assert not ngspice_available()


def test_not_installed_raises_with_installation_instructions(monkeypatch):
    """The whole point of the graceful path: the error tells you what to do."""
    monkeypatch.setattr("virtual_diagnostics.spice.ngspice_executable", lambda: None)
    with pytest.raises(NgspiceNotFound) as caught:
        SpiceFrontEnd(netlist=RC_LOW_PASS).simulate(RC_LOW_PASS)

    message = str(caught.value)
    assert "apt install ngspice" in message
    assert "brew install ngspice" in message
    assert "NGSPICE_EXECUTABLE" in message


def test_a_failing_ngspice_reports_its_own_output(tmp_path, monkeypatch):
    """If ngspice runs but writes no rawfile, its log has to reach the user."""
    fake = tmp_path / "ngspice"
    fake.write_text("#!/bin/sh\necho 'Error: no such node' >&2\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setenv("NGSPICE_EXECUTABLE", str(fake))
    with pytest.raises(NgspiceError, match="no such node"):
        SpiceFrontEnd(netlist=RC_LOW_PASS).simulate(RC_LOW_PASS)


def test_a_hanging_ngspice_times_out(tmp_path, monkeypatch):
    fake = tmp_path / "ngspice"
    fake.write_text("#!/bin/sh\nsleep 30\n")
    fake.chmod(0o755)
    monkeypatch.setenv("NGSPICE_EXECUTABLE", str(fake))
    with pytest.raises(NgspiceError, match="did not finish"):
        SpiceFrontEnd(netlist=RC_LOW_PASS, timeout=1.0).simulate(RC_LOW_PASS)


# -- netlist construction (no ngspice needed) -----------------------------


def test_the_built_netlist_is_inspectable_and_decimated():
    t, v = pulse(points=20_000)
    built = SpiceFrontEnd(netlist=RC_LOW_PASS, max_points=500).build(t - t[0], v)

    assert built.count(".tran") == 1, "the original .tran is replaced, not duplicated"
    assert built.strip().endswith(".end")
    pwl = built[built.index("PWL(") + 4 : built.index(")", built.index("PWL("))]
    assert len(pwl.split()) // 2 <= 500


def test_max_step_is_passed_through():
    t, v = pulse(points=100)
    built = SpiceFrontEnd(netlist=RC_LOW_PASS, max_step=1e-12).build(t - t[0], v)
    tran = [line for line in built.splitlines() if line.startswith(".tran")][0]
    assert len(tran.split()) == 5, "step, end, tstart, tmax"


def test_a_missing_source_is_a_clear_error():
    t, v = pulse()
    with pytest.raises(ValueError, match="no source named"):
        SpiceFrontEnd(netlist=RC_LOW_PASS, sources="Vbeam").build(t - t[0], v)


def test_wrong_number_of_waveforms_is_a_clear_error():
    t, v = pulse()
    front_end = SpiceFrontEnd(netlist=SUMMING, sources=("Va", "Vb"), outputs="out")
    with pytest.raises(ValueError, match="drives 2 sources"):
        front_end.build(t, v)
    with pytest.raises(ValueError, match="unknown sources"):
        front_end.build(t, {"Va": v, "Vc": v})
    with pytest.raises(ValueError, match="no signal given"):
        front_end.build(t, {"Va": v})


def test_a_netlist_can_be_loaded_from_a_file(tmp_path):
    path = tmp_path / "frontend.cir"
    path.write_text(RC_LOW_PASS)
    assert "RC low pass" in SpiceFrontEnd(netlist=str(path)).netlist_text()


# -- end to end, needs the ngspice binary ---------------------------------

needs_ngspice = pytest.mark.skipif(
    not ngspice_available(), reason="needs the ngspice binary (apt install ngspice)"
)


@needs_ngspice
def test_ngspice_version_is_reported():
    assert "ngspice" in ngspice_version().lower()


@needs_ngspice
def test_a_low_pass_attenuates_and_delays():
    t, v = pulse()
    t_out, v_out = SpiceFrontEnd(netlist=RC_LOW_PASS, sources="Vin", outputs="out").run(t, v)

    assert t_out == pytest.approx(t), "the result comes back on the input time axis"
    assert v_out.max() < v.max()
    assert t_out[v_out.argmax()] > t[v.argmax()]


@needs_ngspice
def test_negative_input_times_survive_the_round_trip():
    """SPICE starts at t=0 and discards anything before it. Beam time axes run
    negative, so the record is shifted in and shifted back out."""
    t, v = pulse(start=-1e-9)
    assert t[0] < 0
    t_out, v_out = SpiceFrontEnd(netlist=RC_LOW_PASS).run(t, v)
    assert t_out[0] == pytest.approx(t[0])
    assert v_out[t_out < -0.8e-9].max() < 0.05


@needs_ngspice
def test_gain_is_applied():
    netlist = "* x10\nVin inp 0 PWL(0 0)\nEout out 0 inp 0 10.0\nRload out 0 1e6\n.end\n"
    t, v = pulse()
    _, v_out = SpiceFrontEnd(netlist=netlist).run(t, v)
    assert v_out.max() == pytest.approx(10 * v.max(), rel=0.02)


@needs_ngspice
def test_two_sources_are_driven_in_the_same_run():
    """The reason sources are plural: the netlist has to see them together
    before it can combine them."""
    t, v = pulse()
    front_end = SpiceFrontEnd(netlist=SUMMING, sources=("Va", "Vb"), outputs="out")
    _, both = front_end.run(t, np.stack([v, v]))
    _, only_a = front_end.run(t, np.stack([v, np.zeros_like(v)]))
    assert both.max() == pytest.approx(2 * only_a.max(), rel=0.02)


@needs_ngspice
def test_signals_can_be_given_by_name():
    t, v = pulse()
    front_end = SpiceFrontEnd(netlist=SUMMING, sources=("Va", "Vb"), outputs="out")
    _, by_array = front_end.run(t, np.stack([v, 0.5 * v]))
    _, by_name = front_end.run(t, {"Va": v, "Vb": 0.5 * v})
    assert by_name == pytest.approx(by_array)


@needs_ngspice
def test_a_delay_line_in_the_netlist_produces_two_peaks():
    """The topology this exists for: the near electrode arrives directly, the
    far one arrives a line delay later, on one trace."""
    t, near = pulse(amplitude=1.0)
    _, far = pulse(amplitude=0.6)
    t_out, v = SpiceFrontEnd(netlist=DELAY_LINE, sources=("Va", "Vb"), outputs="out").run(
        t, np.stack([near, far])
    )

    first = t_out[np.flatnonzero(v >= 0.2 * v.max())[0]]
    window = 0.5e-9
    direct = v[np.abs(t_out - first) <= window].max()
    delayed = v[np.abs(t_out - (first + 2e-9)) <= window].max()

    assert direct > 0 and delayed > 0
    assert direct > delayed
    assert delayed / direct == pytest.approx(0.6, rel=0.25)


@needs_ngspice
def test_several_outputs_come_back_as_a_dict():
    t, v = pulse()
    front_end = SpiceFrontEnd(netlist=SUMMING, sources=("Va", "Vb"), outputs=("sum", "out"))
    _, results = front_end.run(t, np.stack([v, v]))
    assert set(results) == {"sum", "out"}
    assert results["out"] == pytest.approx(results["sum"], abs=1e-9)


@needs_ngspice
def test_a_missing_output_node_is_a_clear_error():
    t, v = pulse()
    with pytest.raises(NgspiceError, match="not in the results"):
        SpiceFrontEnd(netlist=RC_LOW_PASS, outputs="nowhere").run(t, v)


@needs_ngspice
def test_button_signals_go_straight_in():
    """ElectrodeSignals.volts is already shaped (n_electrodes, n_samples)."""
    from conftest import gaussian_beam

    from virtual_diagnostics import ButtonBPM

    signals = ButtonBPM(pipe_radius=20e-3).measure(
        gaussian_beam(n=20_000, sigma_x=150e-6, sigma_y=150e-6), rng=0
    )
    _, out = SpiceFrontEnd(netlist=SUMMING, sources=("Va", "Vb"), outputs="out").run(
        signals.t, signals.volts[:2]
    )
    assert out.shape == signals.t.shape
    assert np.isfinite(out).all()


@needs_ngspice
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
        t, v = combiner.run(*(lambda s: (s.t, s.volts))(bpm.measure(beam, rng=1)))
        first = t[np.flatnonzero(v >= 0.2 * v.max())[0]]
        window = 0.5e-9
        direct = v[np.abs(t - first) <= window].max()
        delayed = v[np.abs(t - (first + 2e-9)) <= window].max()
        ratios.append((direct - delayed) / (direct + delayed))
    assert np.all(np.diff(ratios) > 0), "the peak difference tracks position"
