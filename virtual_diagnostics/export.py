"""Getting signals out --- to SPICE, to CSV, to HDF5.

The SPICE path is deliberately just a piecewise-linear source file.  Generating
netlists or driving a simulator would mean owning somebody else's front-end
design; writing the stimulus lets you drop the beam signal into whatever
amplifier, cable model or ADC front end you have already built.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def to_spice_pwl(
    path: str | os.PathLike,
    t: ArrayLike,
    v: ArrayLike,
    shift_to_zero: bool = True,
    precision: int = 9,
) -> NDArray[np.floating]:
    """Write a waveform as a SPICE piecewise-linear source file.

    Use it as the stimulus for your own front-end netlist::

        V1 pickup 0 PWL FILE=ict.pwl
        R1 pickup 0 50
        .tran 10p 200n

    Both ngspice and LTspice accept a two-column whitespace-separated file.

    Parameters
    ----------
    path : path-like
        Destination file.
    t : array_like
        Times in seconds.  Must be increasing.
    v : array_like
        Values in volts (or amps, for a ``PWL`` current source).
    shift_to_zero : bool, optional
        Shift the record so it starts at ``t = 0``.  Beam time axes are centred
        on the bunch and run negative, and a SPICE transient analysis starts at
        zero and simply ignores everything before it --- which silently throws
        away the front of your pulse.  Leave this on unless you have already
        offset the axis yourself.
    precision : int, optional
        Significant figures written.

    Returns
    -------
    ndarray
        The time column as written, so you can line up SPICE results with the
        original beam time axis afterwards.

    Examples
    --------
    >>> import numpy as np, tempfile, os
    >>> t = np.linspace(-1e-9, 1e-9, 5)
    >>> v = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    >>> path = os.path.join(tempfile.mkdtemp(), "pulse.pwl")
    >>> written = to_spice_pwl(path, t, v)
    >>> float(written[0])
    0.0
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if t.shape != v.shape or t.ndim != 1:
        raise ValueError(f"t and v must be matching 1-D arrays, got {t.shape} and {v.shape}")
    if t.size < 2:
        raise ValueError("a PWL source needs at least two points")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    if shift_to_zero:
        t = t - t[0]
    with open(path, "w") as handle:
        for time, value in zip(t, v):
            handle.write(f"{time:.{precision}e}\t{value:.{precision}e}\n")
    return t


def to_csv(
    path: str | os.PathLike,
    header: str = "",
    **columns: ArrayLike,
) -> None:
    """Write named equal-length columns to a CSV file.

    Parameters
    ----------
    path : path-like
        Destination file.
    header : str, optional
        Extra comment text placed above the column names.  The column-name row
        is always comment-prefixed, so :func:`numpy.loadtxt` reads the file back
        with no ``skiprows``.
    **columns : array_like
        Column name to values, e.g. ``to_csv("ict.csv", t=t, volts=v)``.

    Raises
    ------
    ValueError
        If the columns are not all the same length.
    """
    if not columns:
        raise ValueError("nothing to write: pass at least one column")
    arrays = {name: np.asarray(value, dtype=float).ravel() for name, value in columns.items()}
    lengths = {name: array.size for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"columns must be the same length, got {lengths}")
    stacked = np.column_stack(list(arrays.values()))
    np.savetxt(
        path,
        stacked,
        delimiter=",",
        header=(header + "\n" if header else "") + ",".join(arrays),
        comments="# ",
    )


def to_hdf5(path: str | os.PathLike, **datasets: Any) -> None:
    """Write arrays and scalar metadata to an HDF5 file.

    Requires the ``hdf5`` extra.  Arrays become datasets; anything scalar
    becomes a file-level attribute, which is where instrument settings belong so
    they travel with the data.

    Examples
    --------
    >>> to_hdf5("shot.h5", image=image.counts, charge=250e-12)   # doctest: +SKIP
    """
    import h5py

    with h5py.File(path, "w") as handle:
        for name, value in datasets.items():
            if np.isscalar(value):
                handle.attrs[name] = value
            else:
                handle.create_dataset(name, data=np.asarray(value))
