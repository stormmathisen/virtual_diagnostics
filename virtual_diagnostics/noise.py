"""Noise primitives shared by every diagnostic.

Free functions, not a class hierarchy.  Each takes an explicit
:class:`numpy.random.Generator` so that a seeded run is reproducible end to
end --- pass the *same* generator to every diagnostic in a shot and the whole
machine replays identically.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

GAUSSIAN_SHOT_THRESHOLD = 1e6
"""Above this expectation value, shot noise is drawn from a Gaussian.

Poisson sampling is exact but slow and eventually overflows; by a million
counts the Gaussian approximation is correct to better than one part in a
thousand of the standard deviation.
"""


def default_rng(rng: np.random.Generator | int | None = None) -> np.random.Generator:
    """Coerce ``None``, a seed, or a generator into a generator.

    ``None`` gives fresh entropy (a genuinely different shot each call); an
    ``int`` gives a reproducible stream.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def shot(expectation: ArrayLike, rng: np.random.Generator | int | None = None) -> NDArray[np.floating]:
    """Apply counting (Poisson) noise to an expected number of quanta.

    Parameters
    ----------
    expectation : array_like
        Expected number of quanta --- photoelectrons, not digitiser counts.
        Negative entries are treated as zero.
    rng : Generator or int, optional
        Random source.

    Returns
    -------
    ndarray
        A noisy realisation, as floats.

    Notes
    -----
    Quantities above :data:`GAUSSIAN_SHOT_THRESHOLD` use a Gaussian of the same
    mean and variance, which is far faster on megapixel images.
    """
    generator = default_rng(rng)
    lam = np.clip(np.asarray(expectation, dtype=float), 0.0, None)
    out = np.empty_like(lam)
    big = lam > GAUSSIAN_SHOT_THRESHOLD
    if np.any(~big):
        out[~big] = generator.poisson(lam[~big])
    if np.any(big):
        out[big] = generator.normal(lam[big], np.sqrt(lam[big]))
    return out


def read(shape, sigma: float, rng: np.random.Generator | int | None = None) -> NDArray[np.floating]:
    """Zero-mean Gaussian noise, the electronics floor of a sensor or amplifier.

    Parameters
    ----------
    shape : int or tuple of int
        Output shape.
    sigma : float
        Standard deviation, in whatever unit the caller is working in.
    rng : Generator or int, optional
        Random source.
    """
    if sigma <= 0:
        return np.zeros(shape)
    return default_rng(rng).normal(0.0, sigma, shape)


def quantise(
    values: ArrayLike,
    bits: int = 12,
    full_scale: float | None = None,
    dtype=np.uint16,
) -> NDArray:
    """Clip to a full-scale range and round onto an ADC ladder.

    Parameters
    ----------
    values : array_like
        Signal in counts (or volts, if ``full_scale`` is in volts).
    bits : int, optional
        ADC resolution.  The output spans ``0 .. 2**bits - 1``.
    full_scale : float, optional
        Value that maps to the top code.  Defaults to ``2**bits - 1``, i.e. the
        input is already in counts and only needs clipping and rounding.
    dtype : dtype, optional
        Integer output type.

    Returns
    -------
    ndarray
        Integer codes.  Anything at or above ``full_scale`` saturates at the top
        code --- saturation is visible in the data, exactly as on real hardware.
    """
    top = 2**bits - 1
    scale = top if full_scale is None else float(full_scale)
    if scale <= 0:
        raise ValueError(f"full_scale must be positive, got {full_scale}")
    codes = np.rint(np.asarray(values, dtype=float) * (top / scale))
    return np.clip(codes, 0, top).astype(dtype)


def jitter(value: float, sigma: float, rng: np.random.Generator | int | None = None) -> float:
    """Perturb a scalar by a Gaussian, for shot-to-shot machine jitter."""
    if sigma <= 0:
        return float(value)
    return float(default_rng(rng).normal(value, sigma))
