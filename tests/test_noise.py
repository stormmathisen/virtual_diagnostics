"""Noise primitives: the statistics have to be right, and seeds have to hold."""

import numpy as np
import pytest

from virtual_diagnostics.noise import (
    GAUSSIAN_SHOT_THRESHOLD,
    default_rng,
    jitter,
    quantise,
    read,
    shot,
)


def test_shot_noise_has_poisson_statistics():
    sample = shot(np.full(200_000, 25.0), rng=0)
    assert sample.mean() == pytest.approx(25.0, rel=0.01)
    assert sample.var() == pytest.approx(25.0, rel=0.05)
    assert np.all(sample == np.floor(sample)), "Poisson branch must give whole quanta"


def test_shot_noise_switches_to_a_gaussian_when_it_gets_large():
    lam = 10 * GAUSSIAN_SHOT_THRESHOLD
    sample = shot(np.full(20_000, lam), rng=0)
    assert sample.mean() == pytest.approx(lam, rel=1e-3)
    assert sample.std() == pytest.approx(np.sqrt(lam), rel=0.05)


def test_shot_noise_treats_negatives_as_no_signal():
    assert np.all(shot(np.array([-5.0, -1.0, 0.0]), rng=0) == 0.0)


def test_read_noise():
    assert read((1000,), 0.0).max() == 0.0
    assert read((200_000,), 3.0, rng=0).std() == pytest.approx(3.0, rel=0.02)


def test_quantise_clips_at_both_ends():
    codes = quantise([-10.0, 0.0, 2047.5, 1e9], bits=12)
    assert codes[0] == 0
    assert codes[-1] == 4095
    assert codes.dtype == np.uint16


def test_quantise_maps_full_scale_to_the_top_code():
    assert quantise([1.0], bits=12, full_scale=1.0)[0] == 4095
    assert quantise([0.5], bits=12, full_scale=1.0)[0] == pytest.approx(2048, abs=1)


def test_quantise_rejects_a_nonsense_full_scale():
    with pytest.raises(ValueError, match="full_scale"):
        quantise([1.0], full_scale=0.0)


def test_jitter():
    assert jitter(5.0, 0.0) == 5.0
    samples = [jitter(5.0, 0.1, rng=default_rng(0)) for _ in range(10)]
    assert len(set(samples)) == 1, "an explicit generator instance is consumed by the caller"


def test_default_rng_is_reproducible_and_passes_generators_through():
    assert default_rng(3).random() == default_rng(3).random()
    generator = np.random.default_rng(0)
    assert default_rng(generator) is generator
