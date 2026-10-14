"""Create per-scale clean masks by first convolving the image to each scale, which
is the ``convolve_first`` option"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_fill_holes, grey_opening
from scipy.signal import fftconvolve

from eye_patch.erosion import (
    _sort_scales,
    _wsclean_scale_to_sigma,
    create_multi_scale_erosion,
)
from eye_patch.flood_fill import _flood_fill
from eye_patch.logging import logger
from eye_patch.mac import minimum_absolute_clip_threshold
from eye_patch.naming import FITSMaskNames, create_fits_mask_names

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from eye_patch.options import MaskingOptions


def convolve_image_by_scale(
    image_data: NDArray[np.floating], scale: int
) -> NDArray[np.floating]:
    """Generate a convolved version of the input image based on some convolved scale.
    Note that here the scale refers to the scales used by wsclean, which (via radler)
    forms the Gaussian kernel of a scale with:

    >> sigma = 3 / 16 * scale

    which is a FWHM of about 0.44 times the scale.

    Also note that the initial stage of this function is to first apply an 'open' filter,
    which preprocessed the image by using a set of minimum and maximum operations. This
    attempts to separate out emission at different scales before subsequent convolution.
    This feels counter productive, but is done to avoid confusion type effocts with
    convolving scales.

    Args:
        image_data (NDArray[np.floating]): The imaged to be convolved
        scale (int): The pixel scale to smooth with

    Returns:
        NDArray[np.floating]: Smoother version of the input image
    """

    # A grey opening is a minimum then maximum filter, but correctly handles
    # the filter origin so even sized scales do not shift the image
    logger.info("Computing the open filter image")
    image_data = grey_opening(
        input=image_data, size=(scale,) * image_data.ndim, mode="wrap"
    )

    logger.info(f"Convoling with {scale=}")

    # Use the same Gaussian that wsclean forms for this scale, matttee
    sigma = _wsclean_scale_to_sigma(scale=scale)
    fwhm = sigma * 2.0 * np.sqrt(2.0 * np.log(2.0))

    logger.info(f"Generating gaussian kernel for {scale=} {fwhm=:.3f} {sigma=:.3f}")

    # Sample the kernel at unit pixel spacing out to 2.5 sigma either side. An
    # odd number of pixels keeps the kernel centred so the image is not shifted
    half_size = max(1, int(np.ceil(2.5 * sigma)))
    x = np.arange(-half_size, half_size + 1)

    # Form the 2d image grid here, ya sea dog
    distance_squared = x[None, :] ** 2 + x[:, None] ** 2
    kernel = np.exp(-(distance_squared / (2 * sigma**2))).astype(image_data.dtype)

    # Make sure the ships match dimensions
    kernel = kernel.reshape(image_data.shape[:-2] + kernel.shape)
    logger.info(f"{image_data.shape=} {kernel.shape=}")
    image_convolved: NDArray[np.floating] = fftconvolve(image_data, kernel, mode="same")
    return image_convolved


def _create_convolved_scale_mask(
    base_image: NDArray[np.floating],
    fits_header: fits.Header,
    scale: int,
    masking_options: MaskingOptions,
) -> NDArray[np.bool]:
    """Create the clean mask for a single scale of the convolve first approach. The
    image is convolved to the scale, seeded and flood filled using the minimum
    absolute clip, and then eroded by the restoring beam convolved to the scale.

    Args:
        base_image (NDArray[np.floating]): The two-dimensional image at its native resolution. It is not modified.
        fits_header (fits.Header): Header of the image, used to construct the beam shape for the erosion
        scale (int): The scale, in pixels, to construct the clean mask for. A scale of 0 uses the image at its native resolution.
        masking_options (MaskingOptions): Options to use throughout the convolve and erosion process

    Returns:
        NDArray[np.bool]: The clean mask for this scale
    """
    logger.info(f"Processing {scale=}")

    # Image at native scale does not need to be convolved. Nothing below
    # modifies the image in place, so no copy is needed
    convolved_image = (
        convolve_image_by_scale(image_data=base_image, scale=scale)
        if scale > 0
        else base_image
    )

    box_size = masking_options.flood_fill_use_mac_box_size + scale

    # The seed and flood masks share the same base clipping level
    mac_threshold = minimum_absolute_clip_threshold(
        image=convolved_image,
        box_size=box_size,
        adaptive_max_depth=masking_options.flood_fill_use_mac_adaptive_max_depth,
        adaptive_box_step=masking_options.flood_fill_use_mac_adaptive_step_factor,
        adaptive_skew_delta=masking_options.flood_fill_use_mac_adaptive_skew_delta,
    )
    positive_mask = convolved_image > (
        mac_threshold * masking_options.flood_fill_positive_seed_clip
    )

    flood_fill_positive_flood_clip = masking_options.flood_fill_positive_flood_clip
    if scale == 0 and flood_fill_positive_flood_clip < 0.8:
        logger.warning(
            f"{flood_fill_positive_flood_clip=}, too low for {scale=}. Setting to 0.8"
        )
        flood_fill_positive_flood_clip = 0.8

    flood_floor_mask = binary_fill_holes(
        input=convolved_image > (mac_threshold * flood_fill_positive_flood_clip)
    )

    # Ensure positive only pixels. useful (necessary?) at larger scales bridging
    # across negative pixels
    positive_dilated_mask = _flood_fill(
        seed_mask=positive_mask, floor_mask=flood_floor_mask
    )

    scale_mask: NDArray[np.bool] = (
        create_multi_scale_erosion(
            mask=positive_dilated_mask,
            fits_header=fits_header,
            scale=scale,
            minimum_response=masking_options.beam_shape_erode_minimum_response,
            scale_is_fwhm=masking_options.beam_shape_erode_scales_as_fwhm,
        )
        == 1
    )
    return scale_mask


# TODO: This needs some specific unit tests
def create_convolved_erosion_mask(
    fits_image_path: Path,
    masking_options: MaskingOptions,
    overwrite: bool = True,
) -> FITSMaskNames:
    """Create a per-scale clean mask by smoothing the input image to some scale, and then
    applying a reverse flood fill with the minimum absolute clip statistic. After the initial
    mask per scale is constructed a binary erosion is then performed to further refine the mask.

    Args:
        fits_image_path (Path): The path to the FITS image to load.
        masking_options (MaskingOptions): Options to use throughout the convolve and erosion process.
        overwrite (bool): Passed to `fits.writeto`, and will overwrite files should they exist. Defaults to True.

    Returns:
        FITSMaskNames: Clean mask written to a FITS file. Should multiple scales be specified they are encoded bitwise per pixel.
    """
    mask_names = create_fits_mask_names(fits_image=fits_image_path)

    logger.info(f"Getting image data from {fits_image_path=}")
    with fits.open(fits_image_path) as fits_image:
        fits_header = fits_image[0].header
        base_image = fits_image[0].data

    original_shape = base_image.shape

    base_image = np.squeeze(base_image)
    assert len(base_image.shape) == 2, (
        f"The shape of data is {original_shape}, and can not be converted to a two-dimensional image"
    )

    # A single blanked pixel would turn an entire convolved image into NaNs
    non_finite = ~np.isfinite(base_image)
    if np.any(non_finite):
        logger.warning(f"Setting {np.sum(non_finite)} non-finite pixels to zero")
        base_image = np.where(non_finite, 0.0, base_image).astype(base_image.dtype)

    scales = _sort_scales(
        scales=[0]
        if masking_options.beam_shape_erode_scales is None
        else masking_options.beam_shape_erode_scales
    )
    logger.info(f"Will be processing {scales=}")

    output_mask = np.zeros(base_image.shape, dtype=np.float32)

    for index, scale in enumerate(scales):
        scale_mask = _create_convolved_scale_mask(
            base_image=base_image,
            fits_header=fits_header,
            scale=scale,
            masking_options=masking_options,
        )
        output_mask[scale_mask] += 2**index

    logger.info(f"Writing {mask_names.scale_mask_fits}")
    fits.writeto(
        filename=mask_names.scale_mask_fits,
        header=fits_header,
        data=output_mask.reshape(original_shape),
        overwrite=overwrite,
    )
    return FITSMaskNames(scale_mask_fits=mask_names.scale_mask_fits)
