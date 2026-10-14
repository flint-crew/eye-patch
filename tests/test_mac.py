from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import minimum_filter
from scipy.signal import fftconvolve

from eye_patch.mac import (
    create_boxcar_skew_mask,
    minimum_absolute_clip,
    minimum_absolute_clip_threshold,
)

SHAPE = (100, 100)


def test_adaptive_minimum_absolute_clip():
    """Activate the adaptive mode of the mac"""

    image = np.ones((2000, 2000)) * -1.2
    image[100:200, 100:200] = 1.0
    mask = minimum_absolute_clip(
        image=image,
        box_size=50,
        increase_factor=1.1,
        adaptive_max_depth=1,
        adaptive_box_step=4.0,
        adaptive_skew_delta=0.2,
    )
    assert np.all(~mask)

    image = np.ones((2000, 2000)) * -1.2
    image[100:200, 100:200] = 1.0
    image[110, 130] = 2000
    mask = minimum_absolute_clip(
        image=image,
        box_size=50,
        increase_factor=1.1,
        adaptive_max_depth=1,
        adaptive_box_step=4.0,
        adaptive_skew_delta=0.2,
    )
    assert np.sum(mask) == 1


def test_minimum_absolute_clip_simple():
    """The minimum absolute clipping process accepts an image
    and returns a mask.  At its heart it applies a ``minimum_filter``
    from scipy."""

    image = np.ones(SHAPE) * -1.0
    image[10, 10] = -2
    mac_mask = minimum_absolute_clip(image=image, box_size=10)
    assert np.all(~mac_mask)

    image[12, 12] = 5
    mac_mask = minimum_absolute_clip(image=image)
    assert np.sum(mac_mask) == 1

    image[12, 12] = 0
    mac_mask = minimum_absolute_clip(image=image)
    assert np.all(~mac_mask)

    image[12, 12] = 5
    mac_mask = minimum_absolute_clip(image=image, increase_factor=3)
    assert np.all(~mac_mask)


def test_minimum_absolute_clip_threshold() -> None:
    """A shared threshold should reproduce the minimum absolute clip at any factor"""
    image = np.random.default_rng(42).normal(size=(300, 300))
    threshold = minimum_absolute_clip_threshold(image=image, box_size=30)

    rolling_minimum = minimum_filter(input=image, size=30, mode="wrap")
    for increase_factor in (0.8, 1.5, 4.5):
        expected = image > (increase_factor * np.abs(rolling_minimum))
        assert np.array_equal(image > (threshold * increase_factor), expected)
        assert np.array_equal(
            minimum_absolute_clip(
                image=image, increase_factor=increase_factor, box_size=30
            ),
            expected,
        )


@pytest.mark.parametrize("box_size", [24, 25])
def test_create_boxcar_skew_mask_matches_convolution(box_size: int) -> None:
    """The running mean should match a zero padded convolution with a box of ones,
    including the alignment of even sized boxes"""
    image = np.random.default_rng(42).normal(size=(300, 300)) + 0.2

    result = create_boxcar_skew_mask(image=image, skew_delta=0.2, box_size=box_size)

    expected = fftconvolve(
        (image > 0).astype(float), np.ones((box_size, box_size)), mode="same"
    ) / (box_size**2)
    assert np.allclose(result.positive_pixel_frac, np.clip(expected, 0.0, 1.0))
