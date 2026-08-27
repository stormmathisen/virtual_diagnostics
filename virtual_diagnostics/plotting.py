"""Matplotlib helpers for looking at diagnostic output.

Three functions, each returning an :class:`~matplotlib.axes.Axes` and each
accepting ``ax=``, so they compose into your own figures.  There is no style
system and no figure manager --- those belong in your analysis script, not in a
library.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from .screen import ScreenImage


def plot_image(
    image: ScreenImage,
    ax: plt.Axes | None = None,
    units: str = "mm",
    **imshow_kwargs,
) -> plt.Axes:
    """Show a camera frame with physical axes.

    Parameters
    ----------
    image : ScreenImage
    ax : Axes, optional
    units : {"m", "mm", "um"}, optional
        Axis units.
    **imshow_kwargs
        Passed to :func:`matplotlib.pyplot.imshow`.

    Returns
    -------
    Axes
    """
    scale = {"m": 1.0, "mm": 1e3, "um": 1e6}[units]
    if ax is None:
        _, ax = plt.subplots()
    x, y = image.x_axis * scale, image.y_axis * scale
    dx = (x[1] - x[0]) / 2 if x.size > 1 else 0.5
    dy = (y[1] - y[0]) / 2 if y.size > 1 else 0.5
    imshow_kwargs.setdefault("cmap", "inferno")
    imshow_kwargs.setdefault("aspect", "auto")
    mesh = ax.imshow(
        image.counts,
        origin="lower",
        extent=(x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy),
        **imshow_kwargs,
    )
    ax.set_xlabel(f"x ({units})")
    ax.set_ylabel(f"y ({units})")
    ax.figure.colorbar(mesh, ax=ax, label="counts")
    return ax


def plot_projections(
    image: ScreenImage,
    ax: plt.Axes | None = None,
    units: str = "mm",
) -> plt.Axes:
    """Plot both background-subtracted projections of a frame on one axis.

    Each is normalised to its own peak, so a wide dim projection and a narrow
    bright one stay comparable.
    """
    scale = {"m": 1.0, "mm": 1e3, "um": 1e6}[units]
    if ax is None:
        _, ax = plt.subplots()
    ny, nx = image.counts.shape
    for axis, profile, rows, label in (
        (image.x_axis, image.projection_x(), ny, "x"),
        (image.y_axis, image.projection_y(), nx, "y"),
    ):
        signal = profile - image.background * rows
        peak = np.max(np.abs(signal))
        ax.plot(axis * scale, signal / peak if peak > 0 else signal, label=label)
    ax.set_xlabel(f"position ({units})")
    ax.set_ylabel("normalised intensity")
    ax.legend()
    return ax


def plot_signal(
    t: ArrayLike,
    v: ArrayLike,
    ax: plt.Axes | None = None,
    time_units: str = "ns",
    label: str | None = None,
    **plot_kwargs,
) -> plt.Axes:
    """Plot a time-domain waveform from a current monitor or pickup.

    Parameters
    ----------
    t : array_like
        Times in seconds.
    v : array_like
        Values in volts.
    ax : Axes, optional
    time_units : {"s", "ms", "us", "ns", "ps", "fs"}, optional
    label : str, optional
    **plot_kwargs
        Passed to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    Axes
    """
    scale = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9, "ps": 1e12, "fs": 1e15}[time_units]
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(np.asarray(t) * scale, v, label=label, **plot_kwargs)
    ax.set_xlabel(f"time ({time_units})")
    ax.set_ylabel("signal (V)")
    if label is not None:
        ax.legend()
    return ax
