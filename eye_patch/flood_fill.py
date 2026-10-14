"""Flood fill islands seeded from bright pixels, and grow masks to capture faint
and diffuse islands"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import (
    binary_dilation as scipy_binary_dilation,  # Rename to distinguish from skimage
)
from scipy.ndimage import (
    binary_erosion as scipy_binary_erosion,  # Rename to distinguish from skimage
)
from scipy.ndimage import label

from eye_patch.logging import logger
from eye_patch.mac import minimum_absolute_clip_threshold
from eye_patch.snr import _get_signal_image

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from eye_patch.options import MaskingOptions


def _select_large_islands(
    candidate_mask: NDArray[np.bool],
    minimum_island_size: int,
    region_mask: NDArray[np.bool] | None = None,
) -> NDArray[np.bool]:
    """Join nearby candidate pixels with a binary closing, and keep only the islands
    that are made up of at least ``minimum_island_size`` pixels. This is used to find
    regions of faint and diffuse emission that a pixel-wise clip would miss.

    Args:
        candidate_mask (NDArray[np.bool]): Pixels above some low significance level
        minimum_island_size (int): The minimum number of pixels an island should be for it to be kept
        region_mask (NDArray[np.bool] | None, optional): Pixels to remove before the island size constraint is applied. Defaults to None.

    Returns:
        NDArray[np.bool]: Mask of the islands that were kept
    """
    island_mask: NDArray[np.bool] = scipy_binary_dilation(
        input=candidate_mask, iterations=2, structure=np.ones((3, 3))
    )
    island_mask = scipy_binary_erosion(
        input=island_mask, iterations=2, structure=np.ones((3, 3))
    )

    if region_mask is not None:
        island_mask[region_mask] = False

    island_labels, _no_islands = label(input=island_mask, structure=np.ones((3, 3)))

    # Index position is the label, with label 0 being the background
    small_islands = np.bincount(island_labels.ravel()) < minimum_island_size
    small_islands[0] = False
    island_mask[small_islands[island_labels]] = False

    return island_mask


def grow_low_snr_mask(
    image: NDArray[np.floating] | None = None,
    rms: NDArray[np.floating] | None = None,
    background: NDArray[np.floating] | None = None,
    signal: NDArray[np.floating] | None = None,
    grow_low_snr: float = 2.0,
    grow_low_island_size: int = 512,
    region_mask: NDArray[np.bool] | None = None,
) -> NDArray[np.bool]:
    """There may be cases where normal thresholding operations based on simple pixel-wise SNR
    cuts fail to pick up diffuse, low surface brightness regions of emission. When some type
    of masking operation is used there may be instances where these regions are never cleaned.

    Sometimes smoothing can help to pick up these features, but when attempting to pass such
    a mask through to the imagery of choice the islands may be larger than the features at their
    native resolution, unless some other more sophisticated filtering is performed.

    This function attempts to grow masks to capture islands of contiguous pixels above a low
    SNR cut that would otherwise go uncleaned.

    Args:
            Args:
        image (NDArray[np.floating] | None, optional): The total intensity pixels to have the mask for. Defaults to None.
        rms (NDArray[np.floating] | None, optional): The noise across the image. Defaults to None.
        background (NDArray[np.floating] | None, optional): The background acros the image. If None, zeros are assumed. Defaults to None.
        signal(NDArray[np.floating] | None, optional): A signal map. Defaults to None.
        grow_low_snr (float, optional): The SNR pixekls have to be above. Defaults to 2.
        grow_low_island_size (int, optional): The minimum number of pixels an island should be for it to be considered valid. Defaults to 512.
        region_mask (NDArray[np.bool] | None, optional): Whether some region should be masked out before the island size constraint is applied. Defaults to None.

    Returns:
        NDArray[np.bool]: The derived mask of objects with low-surface brightness
    """
    # TODO: The `grow_low_island_size` should be represented in solid angle relative to the restoring beam

    signal = _get_signal_image(
        image=image, rms=rms, background=background, signal=signal
    )

    logger.info(
        f"Growing mask for low surface brightness using {grow_low_snr=} {grow_low_island_size=}"
    )
    return _select_large_islands(
        candidate_mask=signal > grow_low_snr,
        minimum_island_size=grow_low_island_size,
        region_mask=region_mask,
    )


def _flood_fill(
    seed_mask: NDArray[np.bool], floor_mask: NDArray[np.bool]
) -> NDArray[np.bool]:
    """Grow the seed pixels to all connected (8-connectivity) pixels of the floor
    mask.

    Args:
        seed_mask (NDArray[np.bool]): Pixels to grow islands from
        floor_mask (NDArray[np.bool]): Pixels that islands are allowed to grow into

    Returns:
        NDArray[np.bool]: The grown mask, which always includes the seed pixels
    """
    # This is equivalent to a binary dilation of the seed constrained by the
    # floor mask that is iterated until convergence, but is done with a single
    # labelling pass.

    island_labels, no_islands = label(
        input=floor_mask | seed_mask, structure=np.ones((3, 3))
    )

    # Index position is the label, with label 0 being outside of all islands
    seeded_islands = np.zeros(no_islands + 1, dtype=bool)
    seeded_islands[island_labels[seed_mask]] = True
    seeded_islands[0] = False

    flood_mask: NDArray[np.bool] = seeded_islands[island_labels]
    return flood_mask


def _verify_set_positive_seed_clip(
    positive_seed_clip: float, signal: NDArray[np.floating]
) -> float:
    """Ensure that the positive seed clip is handled appropriately. If it
    is too high then it is set to be 90-percent of the maximum signal"""
    # Blanked pixels (e.g. NaNs at the image edge) would otherwise disable this check
    max_signal = np.nanmax(signal)
    if max_signal < positive_seed_clip:
        logger.critical(
            f"The maximum signal {max_signal:.4f} is below the provided {positive_seed_clip=}. "
            "Setting clip to 90 percent of maximum. "
        )
        positive_seed_clip = max_signal * 0.9

    return positive_seed_clip


def create_flood_fill_mask(
    base_image: NDArray[np.floating],
    masking_options: MaskingOptions,
) -> NDArray[np.bool]:
    """Seed islands from bright pixels, and grow them to a lower clipping level.

    Seeds are pixels above ``flood_fill_positive_seed_clip``, and islands grow into
    the connected pixels above ``flood_fill_positive_flood_clip`` (see ``_flood_fill``).
    With ``flood_fill_use_mac`` both clips are increase factors of the minimum
    absolute clip rather than signal-to-noise levels.

    Should ``grow_low_snr_island`` be set, islands of faint and diffuse emission that
    are never seeded are also added, using ``grow_low_snr_island_clip`` and
    ``grow_low_snr_island_size``.

    Args:
        base_image (NDArray[np.floating]): The base image or signal map that is used throughout the fill procedure.
        masking_options (MaskingOptions): Options to carry out masking.

    Returns:
        NDArray[np.bool]: Mask of the pixels to clean
    """

    logger.info("Will be flood filling")
    logger.info(f"{masking_options=} ")

    mac_threshold: NDArray[np.floating] | None = None
    if masking_options.flood_fill_use_mac:
        # The seed, flood and low SNR island masks share the same base clipping level
        mac_threshold = minimum_absolute_clip_threshold(
            image=base_image,
            box_size=masking_options.flood_fill_use_mac_box_size,
            adaptive_max_depth=masking_options.flood_fill_use_mac_adaptive_max_depth,
            adaptive_box_step=masking_options.flood_fill_use_mac_adaptive_step_factor,
            adaptive_skew_delta=masking_options.flood_fill_use_mac_adaptive_skew_delta,
        )
        positive_mask = base_image > (
            mac_threshold * masking_options.flood_fill_positive_seed_clip
        )
        flood_floor_mask = base_image > (
            mac_threshold * masking_options.flood_fill_positive_flood_clip
        )
    else:
        # Sanity check the upper clip level, you rotten seadog
        positive_seed_clip = _verify_set_positive_seed_clip(
            positive_seed_clip=masking_options.flood_fill_positive_seed_clip,
            signal=base_image,
        )
        # Here we create the mask image that will start the binary dilation
        # process, and we will ensure only pixels above the `positive_flood_clip`
        # are allowed to be dilated. In other words we are growing the mask
        positive_mask = base_image >= positive_seed_clip
        flood_floor_mask = base_image > masking_options.flood_fill_positive_flood_clip

    positive_dilated_mask = _flood_fill(
        seed_mask=positive_mask, floor_mask=flood_floor_mask
    )

    if masking_options.grow_low_snr_island:
        if mac_threshold is not None:
            # The base image is not a signal map when the minimum absolute clip is
            # used, so the clip is the increase factor of the same clipping level
            logger.info(
                f"Growing mask for low surface brightness using {masking_options.grow_low_snr_island_clip=} as a minimum absolute clip"
            )
            low_snr_mask = _select_large_islands(
                candidate_mask=base_image
                > (mac_threshold * masking_options.grow_low_snr_island_clip),
                minimum_island_size=masking_options.grow_low_snr_island_size,
            )
        else:
            low_snr_mask = grow_low_snr_mask(
                signal=base_image,
                grow_low_snr=masking_options.grow_low_snr_island_clip,
                grow_low_island_size=masking_options.grow_low_snr_island_size,
            )
        positive_dilated_mask[low_snr_mask] = True

    return positive_dilated_mask


def reverse_negative_flood_fill(
    base_image: NDArray[np.floating],
    masking_options: MaskingOptions,
) -> NDArray[np.bool]:
    """Deprecated, use ``create_flood_fill_mask``. The negative island suppression
    this name refers to has been removed."""
    warnings.warn(
        "reverse_negative_flood_fill is deprecated, use create_flood_fill_mask",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_flood_fill_mask(
        base_image=base_image, masking_options=masking_options
    )
