"""Shared test beam: one well-characterised Gaussian bunch to measure."""

import numpy as np
import pytest

from virtual_diagnostics import Beam

SIGMA_X = 150e-6
SIGMA_Y = 80e-6
SIGMA_T = 1e-12
TOTAL_CHARGE = 250e-12
ENERGY = 100e6


def gaussian_beam(
    n=200_000,
    sigma_x=SIGMA_X,
    sigma_y=SIGMA_Y,
    sigma_t=SIGMA_T,
    total_charge=TOTAL_CHARGE,
    centroid_x=0.0,
    centroid_y=0.0,
    energy=ENERGY,
    relative_energy_spread=0.0,
    seed=0,
):
    """A Gaussian bunch with known moments, for known-answer tests."""
    rng = np.random.default_rng(seed)
    return Beam(
        x=rng.normal(centroid_x, sigma_x, n),
        y=rng.normal(centroid_y, sigma_y, n),
        t=rng.normal(0.0, sigma_t, n),
        px=np.zeros(n),
        py=np.zeros(n),
        pz=energy * (1.0 + rng.normal(0.0, relative_energy_spread, n)),
        q=np.full(n, total_charge / n),
        energy=energy,
    )


@pytest.fixture
def beam():
    return gaussian_beam()


@pytest.fixture
def make_beam():
    return gaussian_beam
