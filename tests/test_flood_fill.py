from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import binary_dilation, label

from eye_patch.flood_fill import (
    _flood_fill,
    _verify_set_seed_clip,
    create_flood_fill_mask,
    grow_low_snr_mask,
)
from eye_patch.options import MaskingOptions


def test_verify_set_seed_clip():
    """Make sure the flood seed clip handles items above all possible values"""
    signal = np.ones((100, 100)) * 10.0

    flood_clip = _verify_set_seed_clip(signal=signal, seed_clip=9.0)

    assert flood_clip == 9.0
    flood_clip = _verify_set_seed_clip(signal=signal, seed_clip=999999.0)
    assert flood_clip == 9.0


def test_create_flood_fill_mask_mac_grows_low_snr_islands() -> None:
    """With the minimum absolute clip the image is not a signal map. Low SNR islands
    should be found using the clip as an increase factor of the same clipping level,
    rather than comparing a significance to the raw pixel values."""
    noise = 1e-4  # An image in Jy/beam rather than a signal map
    yy, xx = np.mgrid[:400, :400]
    diffuse = ((xx - 200) ** 2 + (yy - 200) ** 2) <= 60**2
    image = (
        np.random.default_rng(3).normal(scale=noise, size=(400, 400))
        + diffuse * 2.5 * noise
    )

    without_growth, with_growth = (
        create_flood_fill_mask(
            base_image=image,
            masking_options=MaskingOptions(
                flood_fill=True,
                use_mac=True,
                grow_low_snr_island=grow_low_snr_island,
                grow_low_snr_island_clip=0.5,
            ),
        )
        for grow_low_snr_island in (False, True)
    )

    assert np.mean(without_growth[diffuse]) < 0.1
    assert np.mean(with_growth[diffuse]) > 0.95
    assert np.mean(with_growth[~diffuse]) < 0.02


def test_verify_set_seed_clip_with_nan() -> None:
    """NaNs should not disable the lowering of an unreachable seed clip"""
    signal = np.array([[np.nan, 3.0], [1.0, 2.0]])
    clip = _verify_set_seed_clip(seed_clip=10.0, signal=signal)
    assert np.isclose(clip, 2.7)


@pytest.mark.parametrize("seeds_within_floor", [True, False])
def test_flood_fill_matches_binary_dilation(seeds_within_floor: bool) -> None:
    """The labelling flood fill should match a constrained binary dilation that is
    iterated until convergence, including when seeds sit outside the floor mask"""
    rng = np.random.default_rng(42)
    if seeds_within_floor:
        signal = rng.normal(size=(300, 300))
        seed_mask = signal > 2.5
        floor_mask = signal > 0.3
    else:
        seed_mask = rng.random(size=(300, 300)) > 0.995
        floor_mask = rng.random(size=(300, 300)) > 0.45

    expected = binary_dilation(
        input=seed_mask,
        mask=floor_mask,
        iterations=-1,
        structure=np.ones((3, 3), dtype=bool),
    )
    flood_mask = _flood_fill(seed_mask=seed_mask, floor_mask=floor_mask)

    assert np.array_equal(flood_mask, expected)


def test_grow_low_snr_mask_island_sizes() -> None:
    """Only islands of at least the minimum size should be kept"""
    signal = np.random.default_rng(42).normal(size=(300, 300))
    island_size = 40

    low_snr_mask = grow_low_snr_mask(
        signal=signal, grow_low_snr=0.5, grow_low_island_size=island_size
    )

    island_labels, no_islands = label(input=low_snr_mask, structure=np.ones((3, 3)))
    assert no_islands > 0
    island_counts = np.bincount(island_labels.ravel())[1:]
    assert np.all(island_counts >= island_size)
