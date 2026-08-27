"""The beam contract: moments, energy, and the longitudinal profile."""

import numpy as np
import pytest
from conftest import SIGMA_T, SIGMA_X, SIGMA_Y, TOTAL_CHARGE, gaussian_beam

from virtual_diagnostics import Beam, from_arrays
from virtual_diagnostics.beam import REST_MASS_EV


def test_moments_match_the_generating_distribution(beam):
    assert beam.sigma_x == pytest.approx(SIGMA_X, rel=0.02)
    assert beam.sigma_y == pytest.approx(SIGMA_Y, rel=0.02)
    assert beam.sigma_t == pytest.approx(SIGMA_T, rel=0.02)
    assert beam.total_charge == pytest.approx(TOTAL_CHARGE, rel=1e-12)
    assert beam.centroid_x == pytest.approx(0.0, abs=SIGMA_X / 10)


def test_moments_ignore_the_sign_of_the_charge():
    """Moments weight by |q|, so a code that reports negative charges for
    electrons gets the same beam size rather than sign-flipped nonsense."""
    positive = gaussian_beam(n=10_000)
    negative = Beam(**{**positive.__dict__, "q": -positive.q})
    assert negative.sigma_x == pytest.approx(positive.sigma_x)
    assert negative.total_charge == pytest.approx(-positive.total_charge)


def test_current_profile_conserves_charge(beam):
    t, current = beam.current_profile(bins=400)
    dt = t[1] - t[0]
    assert (current * dt).sum() == pytest.approx(TOTAL_CHARGE, rel=1e-3)


def test_current_profile_honours_an_explicit_dt(beam):
    dt = 1e-13
    t, _ = beam.current_profile(dt=dt, t_range=(-5e-12, 5e-12))
    assert np.diff(t) == pytest.approx(dt, rel=1e-9)


def test_current_profile_rejects_a_backwards_range(beam):
    with pytest.raises(ValueError, match="increasing"):
        beam.current_profile(t_range=(1e-12, -1e-12))


def test_energy_spread(beam):
    spread = gaussian_beam(n=50_000, relative_energy_spread=1e-3)
    assert spread.relative_energy_spread == pytest.approx(1e-3, rel=0.05)
    assert beam.relative_energy_spread == pytest.approx(0.0, abs=1e-9)


def test_relativistic_factors(beam):
    assert beam.relativistic_gamma == pytest.approx(1e8 / REST_MASS_EV["electron"])
    assert beam.relativistic_beta == pytest.approx(1.0, abs=1e-4)
    assert beam.sigma_z == pytest.approx(beam.sigma_t * beam.relativistic_beta * 299792458.0)


def test_shifted_and_scaled(beam):
    moved = beam.shifted(dx=1e-3, dt=5e-12)
    assert moved.centroid_x == pytest.approx(beam.centroid_x + 1e-3)
    assert moved.centroid_t == pytest.approx(beam.centroid_t + 5e-12)
    assert beam.centroid_x == pytest.approx(beam.centroid_x)  # original untouched

    rescaled = beam.scaled_charge(10e-12)
    assert rescaled.total_charge == pytest.approx(10e-12)
    assert rescaled.sigma_x == pytest.approx(beam.sigma_x)


def test_mismatched_array_lengths_are_rejected():
    with pytest.raises(ValueError, match="same shape"):
        from_arrays(x=[0, 1], y=[0], t=[0], px=[0], py=[0], pz=[1e6], q=[1e-12], energy=1e6)


def test_unknown_species_needs_an_explicit_mass():
    kwargs = dict(x=[0.0], y=[0.0], t=[0.0], px=[0.0], py=[0.0], pz=[1e9], q=[1e-12], energy=1e9)
    with pytest.raises(ValueError, match="Unknown species"):
        from_arrays(species="muon", **kwargs)
    assert from_arrays(species="muon", rest_mass=105.66e6, **kwargs).rest_mass == pytest.approx(
        105.66e6
    )
