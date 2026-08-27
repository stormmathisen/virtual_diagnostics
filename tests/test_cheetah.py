"""The Cheetah adapter. Skipped unless the ``cheetah`` extra is installed."""

import numpy as np
import pytest
from conftest import SIGMA_T, SIGMA_X, SIGMA_Y, TOTAL_CHARGE

from virtual_diagnostics import C_LIGHT, from_cheetah
from virtual_diagnostics.beam import REST_MASS_EV

cheetah = pytest.importorskip("cheetah", reason="needs the cheetah extra")
torch = pytest.importorskip("torch")


def cheetah_beam(num_particles=100_000, energy=100e6):
    return cheetah.ParticleBeam.from_parameters(
        num_particles=num_particles,
        sigma_x=torch.tensor(SIGMA_X),
        sigma_y=torch.tensor(SIGMA_Y),
        sigma_tau=torch.tensor(SIGMA_T * C_LIGHT),
        energy=torch.tensor(energy),
        total_charge=torch.tensor(TOTAL_CHARGE),
    )


def test_conversion_preserves_moments():
    source = cheetah_beam()
    beam = from_cheetah(source)
    assert beam.sigma_x == pytest.approx(float(source.sigma_x), rel=1e-5)
    assert beam.sigma_y == pytest.approx(float(source.sigma_y), rel=1e-5)
    assert beam.total_charge == pytest.approx(float(source.total_charge), rel=1e-6)
    assert beam.energy == pytest.approx(float(source.energy), rel=1e-9)
    assert beam.species == "electron"
    # Cheetah stores the mass as float32, hence the loose tolerance.
    assert beam.rest_mass == pytest.approx(REST_MASS_EV["electron"], rel=1e-6)


def test_bunch_length_survives_the_tau_conversion():
    source = cheetah_beam()
    assert from_cheetah(source).sigma_t == pytest.approx(SIGMA_T, rel=0.02)


def test_head_arrives_first():
    """Pin the tau -> t sign convention.

    Cheetah defines z = -beta * tau, so a particle at z > 0 is ahead of the
    reference and must cross a downstream plane *earlier*, i.e. t < 0.  Get
    this backwards and every longitudinal diagnostic is time-reversed,
    silently and without any test noticing.
    """
    particles = torch.zeros(3, 7)
    particles[:, 6] = 1.0
    particles[:, 4] = torch.tensor([-1e-3, 0.0, 1e-3])  # tau, metres
    source = cheetah.ParticleBeam(
        particles=particles,
        energy=torch.tensor(100e6),
        particle_charges=torch.full((3,), TOTAL_CHARGE / 3),
    )

    beam = from_cheetah(source)
    beta = float(source.relativistic_beta)
    z = particles[:, 4].numpy() * -beta

    assert np.all(np.diff(beam.t) > 0), "t must increase with tau"
    assert np.argmax(z) == np.argmin(beam.t), "the particle furthest ahead arrives first"
    assert beam.t == pytest.approx(-z / (beta * C_LIGHT))


def test_momenta_come_back_in_ev_per_c():
    beam = from_cheetah(cheetah_beam(num_particles=2000, energy=250e6))
    assert beam.mean_energy == pytest.approx(250e6, rel=1e-3)
    assert float(np.median(beam.pz)) == pytest.approx(250e6, rel=1e-3)


def test_lost_particles_carry_no_charge():
    source = cheetah_beam(num_particles=1000)
    survival = source.survival_probabilities.clone()
    survival[:500] = 0.0
    source.survival_probabilities = survival

    beam = from_cheetah(source)
    assert beam.total_charge == pytest.approx(TOTAL_CHARGE / 2, rel=1e-6)
    assert np.all(beam.q[:500] == 0.0)


def test_batched_beams_are_rejected():
    batched = cheetah.ParticleBeam.from_parameters(
        num_particles=100,
        sigma_x=torch.tensor([1e-4, 2e-4]),
        energy=torch.tensor(100e6),
        total_charge=torch.tensor(TOTAL_CHARGE),
    )
    with pytest.raises(ValueError, match="unbatched"):
        from_cheetah(batched)


def test_tracking_through_a_lattice_then_imaging():
    """End to end: Cheetah drift changes the beam size, the screen sees it."""
    from virtual_diagnostics import Screen

    source = cheetah.ParticleBeam.from_twiss(
        num_particles=100_000,
        beta_x=torch.tensor(5.0),
        beta_y=torch.tensor(5.0),
        emittance_x=torch.tensor(1e-9),
        emittance_y=torch.tensor(1e-9),
        energy=torch.tensor(100e6),
        total_charge=torch.tensor(TOTAL_CHARGE),
    )
    downstream = cheetah.Drift(length=torch.tensor(2.0)).track(source)
    beam = from_cheetah(downstream)

    # A quiet screen: this test is about geometry surviving the round trip, not
    # about noise, and the single-shot RMS estimator is noise-limited in the
    # tails (see test_screen.py::test_measured_size_is_unbiased_over_many_shots).
    screen = Screen(
        pixel_size=10e-6,
        resolution=(400, 400),
        counts_per_pc=1e4,
        dark_offset=0.0,
        read_noise=0.0,
        bits=16,
    )
    image = screen.measure(beam, rng=0)
    assert image.off_screen_fraction < 0.01
    assert image.saturated_fraction == 0.0
    assert image.beam_size()[0] == pytest.approx(beam.sigma_x, rel=0.05)
    assert image.beam_size()[1] == pytest.approx(beam.sigma_y, rel=0.05)
