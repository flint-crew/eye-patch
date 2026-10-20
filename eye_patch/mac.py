"""The minimum absolute clip statistic, including the boxcar skew test used by
its adaptive mode"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.ndimage import minimum_filter, uniform_filter

from eye_patch.logging import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SkewResult(NamedTuple):
    positive_pixel_frac: NDArray[np.floating]
    """The fraction of positive pixels in a boxcar function"""
    skew_mask: NDArray[np.bool]
    """Mask of pixel positions indicating which positions failed the skew test"""
    box_size: int
    """Size of the boxcar window applies"""
    skew_delta: float
    """The test threshold for skew"""


def create_boxcar_skew_mask(
    image: NDArray[np.floating], skew_delta: float, box_size: int
) -> SkewResult:
    """Compute the fraction of positive pixels within a rolling boxcar, and flag
    the positions where this fraction exceeds ``0.5 + skew_delta``. For zero mean
    noise there should be a similar number of positive and negative pixels, so an
    excess of positive pixels suggests the boxcar is too small relative to some
    feature. Only the sign of each pixel is considered, so bright pixels do not
    contribute more than faint ones.

    Args:
        image (NDArray[np.floating]): The two-dimensional image to consider
        skew_delta (float): The allowed deviation away from 0.5 before a position is considered skewed. Should be between 0 and 0.5 (exclusive).
        box_size (int): The size of the boxcar in pixels

    Returns:
        SkewResult: The positive pixel fraction and the mask of skewed positions
    """
    assert 0.0 < skew_delta < 0.5, f"{skew_delta=}, but should be 0.0 to 0.5"
    assert len(image.shape) == 2, (
        f"Expected two dimensions, got image shape of {image.shape}"
    )
    logger.info(f"Computing boxcar skew with {box_size=} and {skew_delta=}")

    # Counting positive pixel fraction here, a running mean on
    # boolean mask
    positive_pixel_fraction = uniform_filter(
        input=image > 0.0,
        size=box_size,
        output=np.float64,
        mode="constant",
        cval=0.0,
    )
    positive_pixel_fraction = np.clip(
        positive_pixel_fraction, 0.0, 1.0
    )  # trust nothing

    skew_mask = positive_pixel_fraction > (0.5 + skew_delta)
    logger.info(f"{np.sum(skew_mask)} pixels above {skew_delta=} with {box_size=}")

    return SkewResult(
        positive_pixel_frac=positive_pixel_fraction,
        skew_mask=skew_mask,
        skew_delta=skew_delta,
        box_size=box_size,
    )


def _adaptive_rolling_minimum(
    image: NDArray[np.floating],
    box_size: int = 100,
    adaptive_max_depth: int = 3,
    adaptive_box_step: float = 2.0,
    adaptive_skew_delta: float = 0.2,
) -> NDArray[np.floating]:
    """Derive the rolling boxcar minimum where the box size is adaptively scaled
    if an excess of positive pixels is detected towards some direction. Should a
    set of pixels fail the skewness test then a larger box is used and the minimum
    from that larger box is used.

    Args:
        image (NDArray[np.floating]): The input image to compute the adaptive rolling minimum against
        box_size (int, optional): The size of the box in pixels. Defaults to 100.
        adaptive_max_depth (int, optional): The maximum number of times the box can be enlarged. Defaults to 3.
        adaptive_box_step (float, optional): The factor the box is increased by for pixels that failed the skewness test. Defaults to 2.0.
        adaptive_skew_delta (float, optional): The deviation away from 0.5 that is allowed for positive / (positive + negative) before a region is considered skewed. Defaults to 0.2.

    Returns:
        NDArray[np.floating]: The rolling minimum
    """
    logger.info(
        f"Using adaptive minimum absolute clip with {box_size=} {adaptive_skew_delta=}"
    )
    min_value: NDArray[np.floating] = minimum_filter(
        input=image, size=box_size, mode="wrap"
    )

    for box_round in range(adaptive_max_depth, 0, -1):
        skew_results = create_boxcar_skew_mask(
            image=image, skew_delta=adaptive_skew_delta, box_size=box_size
        )
        if np.all(~skew_results.skew_mask):
            logger.info("No skewed islands detected")
            break
        if any(box_size > dim for dim in image.shape):
            logger.info(f"{box_size=} larger than a dimension in {image.shape=}")
            break

        logger.info(f"({box_round}) Growing {box_size=} {adaptive_box_step=}")
        box_size = int(box_size * adaptive_box_step)
        _min_value = minimum_filter(input=image, size=box_size, mode="wrap")
        logger.debug("Slicing minimum values into place")

        min_value[skew_results.skew_mask] = _min_value[skew_results.skew_mask]

    return min_value


def minimum_absolute_clip_threshold(
    image: NDArray[np.floating],
    box_size: int = 100,
    adaptive_max_depth: int | None = None,
    adaptive_box_step: float = 2.0,
    adaptive_skew_delta: float = 0.2,
) -> NDArray[np.floating]:
    """Compute the absolute of the (optionally adaptive) rolling boxcar minimum,
    which is the base clipping level of the minimum absolute clip before it is
    increased by some guard factor. Computing it once allows a single image to
    be clipped at multiple guard factors without repeating the expensive filtering.

    Args:
        image (NDArray[np.floating]): Image to compute the clipping level for
        box_size (int, optional): Size of the box car of the minimum filter. Defaults to 100.
        adaptive_max_depth (int | None, optional): The maximum number of rounds that the adaptive mode is allowed to perform when rescaling boxcar results in certain directions. If None the adaptive mode is not used. Defaults to None.
        adaptive_box_step (float, optional): A multiplicative factor to increase the boxcar size by each round. Defaults to 2.0.
        adaptive_skew_delta (float, optional): Minimum deviation from 0.5 that needs to be met to classify a region as skewed. Defaults to 0.2.

    Returns:
        NDArray[np.floating]: The absolute rolling minimum
    """
    if adaptive_max_depth is None:
        logger.info(f"Minimum absolute clip threshold, {box_size=}")
        rolling_min = minimum_filter(input=image, size=box_size, mode="wrap")
    else:
        rolling_min = _adaptive_rolling_minimum(
            image=image,
            box_size=box_size,
            adaptive_max_depth=int(adaptive_max_depth),
            adaptive_box_step=adaptive_box_step,
            adaptive_skew_delta=adaptive_skew_delta,
        )

    threshold: NDArray[np.floating] = np.abs(rolling_min)
    return threshold


def minimum_absolute_clip(
    image: NDArray[np.floating],
    increase_factor: float = 2.0,
    box_size: int = 100,
    adaptive_max_depth: int | None = None,
    adaptive_box_step: float = 2.0,
    adaptive_skew_delta: float = 0.2,
) -> NDArray[np.bool]:
    """Implements minimum absolute clip method. A minimum filter of a particular
    boxc size is applied to the input image. The absolute of the output is taken
    and increased by a guard factor, which forms the clipping level used to construct
    a clean mask:

    >>> image > (absolute(minimum_filter(image, box)) * factor)

    The idea is only valid for zero mean and normally distributed pixels, with
    positive definite flux, making it appropriate for Stokes I.

    Larger box sizes and guard factors will make the mask more conservative. Should
    the boxcar be too small relative to some feature it is aligned it is possible
    that an excess of positive pixels will produce an less than optimal clipping
    level. An adaptive box size mode, if activated, attempts to use a larger box
    around these regions.

    The basic idea being detecting regions where the boxcar is too small is around
    the idea that there should be a similar number of positive to negative pixels.
    Should there be too many positive pixels in a region it is likely there is an

    Args:
        image (NDArray[np.floating]): Image to create a mask for
        increase_factor (float, optional): The guard factor used to inflate the absolute of the minimum filter. Defaults to 2.0.
        box_size (int, optional): Size of the box car of the minimum filter. Defaults to 100.
        adaptive_max_depth (int | None, optional): The maximum number of rounds that the adaptive mode is allowed to perform when rescaling boxcar results in certain directions. Defaults to None.
        adaptive_box_step (float, optional): A multiplicative factor to increase the boxcar size by each round. Defaults to 2.0.
        adaptive_skew_delta (float, optional): Minimum deviation from 0.5 that needs to be met to classify a region as skewed. Defaults to 0.2.

    Returns:
        NDArray[np.bool]: Final mask
    """

    threshold = minimum_absolute_clip_threshold(
        image=image,
        box_size=box_size,
        adaptive_max_depth=adaptive_max_depth,
        adaptive_box_step=adaptive_box_step,
        adaptive_skew_delta=adaptive_skew_delta,
    )
    logger.info(f"Minimum absolute clip, {increase_factor=} {box_size=}")
    mac_mask: NDArray[np.bool] = image > (threshold * increase_factor)
    return mac_mask
