"""Plotting: the helpers have to accept an axis and label the physical units."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from virtual_diagnostics import (  # noqa: E402
    CurrentMonitor,
    Screen,
    plot_image,
    plot_projections,
    plot_signal,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_image_uses_physical_axes(beam):
    image = Screen(pixel_size=10e-6, resolution=(100, 80)).measure(beam, rng=0)
    ax = plot_image(image, units="mm")
    assert ax.get_xlabel() == "x (mm)"
    left, right = ax.get_xlim()
    assert right - left == pytest.approx(100 * 10e-6 * 1e3, rel=1e-6)


def test_plot_helpers_accept_an_existing_axis(beam):
    image = Screen(pixel_size=10e-6, resolution=(60, 60)).measure(beam, rng=0)
    _, axes = plt.subplots(1, 3)
    assert plot_image(image, ax=axes[0]) is axes[0]
    assert plot_projections(image, ax=axes[1]) is axes[1]
    t, v = CurrentMonitor().measure(beam, rng=0)
    assert plot_signal(t, v, ax=axes[2], label="ICT") is axes[2]


def test_plot_signal_scales_the_time_axis():
    t = np.linspace(0, 1e-9, 10)
    ax = plot_signal(t, np.zeros(10), time_units="ns")
    assert ax.get_xlabel() == "time (ns)"
    assert ax.lines[0].get_xdata().max() == pytest.approx(1.0)
