"""Export: the files have to be readable by the tools they are written for."""

import numpy as np
import pytest

from virtual_diagnostics import to_csv, to_spice_pwl


def waveform():
    t = np.linspace(-1e-9, 3e-9, 401)
    return t, np.exp(-((t / 5e-10) ** 2))


def test_pwl_round_trips_through_the_file(tmp_path):
    t, v = waveform()
    path = tmp_path / "pulse.pwl"
    written = to_spice_pwl(path, t, v)

    columns = np.loadtxt(path)
    assert columns.shape == (t.size, 2)
    assert columns[:, 0] == pytest.approx(written, rel=1e-8)
    assert columns[:, 1] == pytest.approx(v, rel=1e-8)


def test_pwl_shifts_the_record_to_start_at_zero(tmp_path):
    """SPICE transient analysis starts at t=0 and silently discards anything
    before it, which would eat the front of a beam-centred record."""
    t, v = waveform()
    written = to_spice_pwl(tmp_path / "shifted.pwl", t, v)
    assert written[0] == 0.0
    assert written[-1] == pytest.approx(t[-1] - t[0])

    kept = to_spice_pwl(tmp_path / "raw.pwl", t, v, shift_to_zero=False)
    assert kept[0] == pytest.approx(t[0])


def test_pwl_rejects_records_spice_cannot_use(tmp_path):
    t, v = waveform()
    with pytest.raises(ValueError, match="increasing"):
        to_spice_pwl(tmp_path / "a.pwl", t[::-1], v)
    with pytest.raises(ValueError, match="two points"):
        to_spice_pwl(tmp_path / "b.pwl", t[:1], v[:1])
    with pytest.raises(ValueError, match="matching"):
        to_spice_pwl(tmp_path / "c.pwl", t, v[:-1])


def test_csv_writes_named_columns(tmp_path):
    t, v = waveform()
    path = tmp_path / "signal.csv"
    to_csv(path, t=t, volts=v)

    assert path.read_text().splitlines()[0] == "# t,volts"
    columns = np.loadtxt(path, delimiter=",")
    assert columns[:, 0] == pytest.approx(t)
    assert columns[:, 1] == pytest.approx(v)


def test_csv_validates_its_input(tmp_path):
    t, v = waveform()
    with pytest.raises(ValueError, match="same length"):
        to_csv(tmp_path / "a.csv", t=t, volts=v[:-1])
    with pytest.raises(ValueError, match="at least one column"):
        to_csv(tmp_path / "b.csv")


def test_hdf5_splits_arrays_from_metadata(tmp_path):
    h5py = pytest.importorskip("h5py", reason="needs the hdf5 extra")
    from virtual_diagnostics import to_hdf5

    t, v = waveform()
    path = tmp_path / "shot.h5"
    to_hdf5(path, time=t, volts=v, charge=250e-12, monitor="ICT-S04")

    with h5py.File(path) as handle:
        assert handle["time"][:] == pytest.approx(t)
        assert handle["volts"][:] == pytest.approx(v)
        assert handle.attrs["charge"] == pytest.approx(250e-12)
        assert handle.attrs["monitor"] == "ICT-S04"
