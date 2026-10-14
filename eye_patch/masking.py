"""Utility functions to make image based masks from images, with the initial
thought being towards FITS images.
"""

from __future__ import annotations

import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import NamedTuple, TypeAlias

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from capn_crunch import BaseOptions, add_options_to_parser, create_options_from_parser
from numpy.typing import NDArray
from radio_beam import Beam
from reproject import reproject_interp
from scipy.ndimage import (
    binary_dilation as scipy_binary_dilation,  # Rename to distinguish from skimage
)
from scipy.ndimage import (
    binary_erosion as scipy_binary_erosion,  # Rename to distinguish from skimage
)
from scipy.ndimage import (
    binary_fill_holes,
    grey_opening,
    label,
    minimum_filter,
    uniform_filter,
)
from scipy.signal import fftconvolve

from eye_patch.logging import logger
from eye_patch.naming import FITSMaskNames, create_fits_mask_names

# Add explicit export so mypy on tests is ok
__all__ = ["create_options_from_parser"]

# TODO: Need to remove a fair amount of old approaches, and deprecate some of the toy functions

# Masks that may be written to a FITS file. Booleans are deliberately excluded, as
# fits.writeto raises on a boolean array, so type checking stops one reaching it.
MaskLike: TypeAlias = NDArray[np.floating]
# Masks that are accepted as either booleans or floats, e.g. by the erosion helpers
FloatOrBoolMask: TypeAlias = NDArray[np.floating] | NDArray[np.bool]


class MaskingOptions(BaseOptions):
    """Contains options for the creation of clean masks from some subject
    image. Clipping levels specified are in units of RMS (or sigma). They
    are NOT in absolute units.
    """

    base_snr_clip: float = 4
    """A base clipping level to be used should other options not be activated"""
    flood_fill: bool = False
    """Whether to attempt to flood fill when constructing a mask. This should be `True` for ``grow_low_snr_island`` to have an effect."""
    flood_fill_positive_seed_clip: float = 4.5
    """The clipping level to seed islands that will be grown to lower signal metric"""
    flood_fill_positive_flood_clip: float = 1.5
    """Clipping level used to grow seeded islands down to"""
    flood_fill_use_mac: bool = False
    """If True, the clipping levels (including ``grow_low_snr_island_clip``) are used as the `increase_factor` when using a minimum absolute clip"""
    flood_fill_use_mac_box_size: int = 75
    """The size of the mac box size should mac be used"""
    flood_fill_use_mac_adaptive_step_factor: float = 2.0
    """Stepping size used to increase box by should adaptive detect poor boxcar statistics"""
    flood_fill_use_mac_adaptive_skew_delta: float = 0.2
    """A box is consider too small for a pixel if the fractional proportion of positive pixels is larger than the deviation away of (0.5 + frac). This threshold is therefore 0 to 0.5"""
    flood_fill_use_mac_adaptive_max_depth: int | None = None
    """Determines the number of adaptive boxcar scales to use when constructing seed mask. If None no adaptive boxcar sizes"""
    grow_low_snr_island: bool = False
    """Whether to attempt to grow a mask to capture islands of low SNR (e.g. diffuse emission)"""
    grow_low_snr_island_clip: float = 1.75
    """The minimum significance levels of pixels to be to seed low SNR islands for consideration. When ``flood_fill_use_mac`` is True this is the `increase_factor` of the minimum absolute clip instead"""
    grow_low_snr_island_size: int = 768
    """The number of pixels an island has to be for it to be accepted"""
    beam_shape_erode: bool = False
    """Erode the mask using the shape of the restoring beam"""
    beam_shape_erode_minimum_response: float = 0.6
    """The minimum response of the beam that is used to form the erode structure shape"""
    beam_shape_erode_scales: tuple[int, ...] | None = None
    """The wsclean multi-scale scales, in pixels, to perform a binary erosion with. Each is converted to the Gaussian that wsclean uses for it (a FWHM of about 0.44 of the scale). Output scales are encoded as a bitmapped value (e.g. n'th scale is n'th bit). Scales are sorted into ascending order to match wsclean"""
    beam_shape_erode_scales_as_fwhm: bool = False
    """Use each of the ``beam_shape_erode_scales`` as the FWHM of the Gaussian, rather than the Gaussian wsclean uses for it. This was the behaviour of earlier versions, and is about 2.3 times wider than the wsclean Gaussian of a scale."""
    convolve_first: bool = False
    """Attempt to construct mask across scales by first convolving the input image by a scale kernel, then run the island construction stage"""


def create_beam_mask_kernel(
    fits_header: fits.Header,
    kernel_size: int | tuple[int, int] = 100,
    minimum_response: float = 0.6,
    pixel_scale: float | None = None,
    auto_resize: bool = True,
) -> NDArray[np.bool]:
    """Make a mask using the shape of a beam in a FITS Header object. The
    beam properties in the ehader are used to generate the two-dimensional
    Gaussian main lobe, from which a cut is made based on the minimum
    power.

    Args:
        fits_header (fits.Header): The FITS header to create beam from
        kernel_size (int | tuple[int, int], optional): Size of the output kernel in pixels. Will be a square. An odd size centres the beam on a pixel, whereas an even size places it between pixels, which shifts any erosion using the kernel by half a pixel. Defaults to 100.
        minimum_response (float, optional): Minimum response of the beam shape for the mask to be constructed from. Defaults to 0.6.
        pixel_scale (float | None, optional): The FWHM, in pixels, of a circular Gaussian that the restoring beam is convolved with. This is not a wsclean multi-scale scale, which should first be converted to its FWHM (see ``create_multi_scale_erosion``). Defaults to None.
        auto_resize (bool, optional): If True and the kernel is evaluated to all 1's, enlarge the ``kernel_size`` to ``2 * kernel_size + 1``, which keeps the kernel size odd. Defaults to True.

    Raises:
        KeyError: Raised if CDELT1 and CDELT2 missing

    Returns:
        NDArray[np.bool]: Boolean mask of the kernel shape
    """
    assert 0.0 < minimum_response < 1.0, (
        f"{minimum_response=}, should be between 0 to 1 (exclusive)"
    )

    POSITION_KEYS = ("CDELT1", "CDELT2")
    if not all(key in fits_header for key in POSITION_KEYS):
        msg = f"{POSITION_KEYS=}  all need to be present"
        raise KeyError(msg)

    beam = Beam.from_fits_header(fits_header)
    assert isinstance(beam, Beam)

    cdelt1, cdelt2 = np.abs(fits_header["CDELT1"]), np.abs(fits_header["CDELT2"])
    assert np.isclose(cdelt1, cdelt2), (
        f"Pixel scales {cdelt1=} {cdelt2=}, but must be equal"
    )

    pixel_unit = u.Unit("deg")

    if pixel_scale is not None:
        assert pixel_scale >= 0.0, f"{pixel_scale=}, but should not be negative"
        logger.debug(f"Convoling {beam=} with {pixel_scale=}")
        scale_fwhm = float(pixel_scale) * cdelt1 * pixel_unit
        scale_beam = Beam(scale_fwhm, scale_fwhm, 0 * u.deg)

        logger.debug(f"Convolving with {scale_beam=}")
        beam = beam.convolve(scale_beam)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    # Trust no one, mate
    assert len(kernel_size) == 2, (
        f"{kernel_size=}, but should be a tuple[int, int] by now"
    )

    max_resize_attempts = 10
    for _ in range(max_resize_attempts):
        y_kernel_size, x_kernel_size = kernel_size

        # NOTE: This could be slow for large sizes.
        # TODO: Consider evaluating the inner, padding the outer?
        k = beam.as_kernel(
            pixscale=cdelt1 * pixel_unit, x_size=x_kernel_size, y_size=y_kernel_size
        )

        mask = k.array > (np.max(k.array) * minimum_response)
        if not auto_resize:
            break

        # Ensure that there be a border of empty pixels, mate. This
        # way we be sure ye larger enough to contain the beam response.
        if (
            np.all(mask[0, :] == 0)
            and np.all(mask[-1, :] == 0)
            and np.all(mask[:, 0] == 0)
            and np.all(mask[:, -1] == 0)
        ):
            break

        # Doubling plus one keeps an odd sized kernel centred on a pixel
        kernel_size = tuple(ks * 2 + 1 for ks in kernel_size)  # type: ignore[assignment]
        logger.warning(f"Increasing to {kernel_size=} for {pixel_scale=}")
    else:
        logger.warning(
            f"{kernel_size=} appears too small for {pixel_scale=} after {max_resize_attempts=}"
        )

    assert isinstance(mask, np.ndarray), f"Expected a numpy array, have {type(mask)}"
    return mask


def fft_binary_erosion(
    mask: FloatOrBoolMask,
    kernel: NDArray[np.bool],
) -> NDArray[np.bool]:
    """Attempt to perform a binary erosion using convolution therom. FFT
    the input mask, FFT the input kernel, multiply, FFT the result. This
    would give the running sum of the kernel moved across the image, which
    in the case of boolean-type values will be the number of overlapping 1's.

    The returned mask is after ensuring the kernel total sum matches the running
    sum.

    Args:
        mask (FloatOrBoolMask): The mask that will be eroded
        kernel (NDArray[np.bool]): The kernel structure used for the erosion

    Returns:
        NDArray[np.bool]: The eroded mask
    """

    logger.info("Beginning fft erosion ... ")
    original_shape = mask.shape

    mask = mask[..., :, :].astype(float)
    kernel = kernel[..., :, :].astype(float)

    erode_sum = np.floor(np.abs(fftconvolve(mask, kernel, mode="same")) + 0.5)

    logger.info("... finished erosion")

    erosion_mask: NDArray[np.bool] = erode_sum >= np.sum(kernel)
    return np.reshape(erosion_mask, original_shape)


def _wsclean_scale_to_sigma(scale: float) -> float:
    """wsclean (via radler) forms the Gaussian of a multi-scale scale with a sigma
    of 3/16 of the scale, which is a FWHM of about 0.44 of the scale"""
    return scale * 3.0 / 16.0


def _sort_scales(scales: list[int] | tuple[int, ...]) -> list[int]:
    """Sort scales into ascending order, warning should their order change.

    wsclean sorts its multi-scale scales and reads the n'th bit of a per-scale
    mask as its n'th smallest scale, so bits must be assigned in ascending order.
    """
    sorted_scales = sorted(scales)
    if sorted_scales != list(scales):
        logger.warning(
            f"Scales {scales} are not in ascending order. Using {sorted_scales} so the n'th bit is the n'th smallest scale, as expected by wsclean"
        )
    return sorted_scales


def _has_beam_parameters(fits_header: fits.Header) -> bool:
    """Whether the header carries the beam keywords needed to construct a beam kernel"""
    return all(key in fits_header for key in ["BMAJ", "BMIN", "BPA"])


def create_multi_scale_erosion(
    mask: FloatOrBoolMask,
    fits_header: fits.Header,
    scale: int,
    minimum_response: float = 0.6,
    scale_is_fwhm: bool = False,
) -> NDArray[np.bool]:
    """Specialised encode per-scale clean masks as bitmapped per-pixel integer values.

    Args:
        mask (FloatOrBoolMask): The current mask that will be eroded based on the beam shape
        fits_header (fits.Header): The fits header of the mask used to generate the beam kernel shape
        scale (int): The wsclean multi-scale scale, in pixels. The restoring beam is convolved by the Gaussian that wsclean uses for this scale before the erosion is performed.
        minimum_response (float, optional): The minimum response of the main restoring beam to craft the shape from. Defaults to 0.6.
        scale_is_fwhm (bool, optional): Use the scale as the FWHM of the Gaussian instead, as in earlier versions. This is about 2.3 times wider than the wsclean Gaussian of the scale, and is intended for comparisons. Defaults to False.

    Returns:
        NDArray[np.bool]: The eroded mask
    """
    # Trust no one on the high seas, mate
    if not _has_beam_parameters(fits_header=fits_header):
        logger.warning(
            "Beam parameters missing. Not performing the beam shape erosion. "
        )
        return mask  # type: ignore[return-value]

    # The scale is not a FWHM. wsclean uses a Gaussian with a sigma of 3/16 of the
    # scale, so a scale of 16 pixels is a Gaussian with a FWHM of about 7 pixels
    scale_fwhm = _wsclean_scale_to_sigma(scale=scale) * 2.0 * np.sqrt(2.0 * np.log(2.0))
    # A scale of 0 is the restoring beam alone either way, so needs no warning
    if scale_is_fwhm and scale > 0:
        logger.warning(
            f"Using {scale=} as the FWHM, as in earlier versions, rather than the {scale_fwhm=:.2f} of the Gaussian wsclean uses"
        )
        scale_fwhm = float(scale)

    logger.info(
        f"Eroding with {scale=} ({scale_fwhm=:.2f} pixels), {minimum_response=}"
    )

    # The max() is to avoid excessive resize warnings at low scales. The
    # kernel size is kept odd so the beam is centred on a pixel, correcting a half
    # pixel shift
    kernel_size = max(17, int(1 + scale * 1.1))
    if kernel_size % 2 == 0:
        kernel_size += 1

    beam_mask_kernel = create_beam_mask_kernel(
        fits_header=fits_header,
        minimum_response=minimum_response,
        pixel_scale=scale_fwhm,
        kernel_size=kernel_size,
        auto_resize=True,
    )

    # This handles any unsqueezed dimensions
    beam_mask_kernel = beam_mask_kernel.reshape(
        mask.shape[:-2] + beam_mask_kernel.shape
    )

    # As far as this pirate knows the scipy_binary_erosion used
    # at the bottom is evaluated in the image plane. For larger
    # images and larger kernels (particularly the later), the cpu
    # cost ballons. Switching to FFT also speeds this up for
    # larger images/kernels.
    # TODO: Refine when the fft mode is used. scipy's memory is a table that
    # depends on kernel size. Cropping the kernel to its structure would also shrink
    # the scipy table by 8x without changing the erosion. The FFT approach is much
    # faster for large kernels, but its memory grows with image size.
    size_ratio = np.prod(beam_mask_kernel.shape) / np.prod(mask.shape)
    logger.debug(f"{size_ratio=}")
    fft_erosion_mode = (size_ratio) > 5e-5

    if fft_erosion_mode:
        logger.debug(
            "Will be using fft-based binary erosion, based on size of kernel-to-mask"
        )
        return fft_binary_erosion(mask=mask, kernel=beam_mask_kernel)

    eroded_mask: NDArray[np.bool] = scipy_binary_erosion(
        input=mask, iterations=1, structure=beam_mask_kernel
    )
    return eroded_mask


def beam_shape_erode(
    mask: MaskLike,
    fits_header: fits.Header,
    minimum_response: float = 0.6,
    scales: list[int] | tuple[int, ...] | None = None,
    scales_as_fwhm: bool = False,
) -> MaskLike:
    """Construct a kernel representing the shape of the restoring beam at
        a particular level, and use it as the basis of a binary erosion of the
        input mask.

        The ``fits_header`` is used to construct the beam shape that matches the
        same pixel size

        The ``scales`` parameters outlines the scales being used to clean at during
        some multi-scale deconvolution algorithm. If provided the binary eroision is
        carried out at each scale (the beam is convolved by a circular gaussian at
        this scale). The outputs are stored as a bitmask per pixel where the n'th scales
        is stored as the n'th bit.

        Args:
            mask (MaskLike): The current mask that will be eroded based on the beam shape
            fits_header (fits.Header): The fits header of the mask used to generate the beam kernel shape
            minimum_response (float, optional): The minimum response of the main restoring beam to craft the shape from. Defaults to 0.6.
            scales (list[int] | tuple[int, ...] | None, optional): Defines the scales that are being used during multi-scale clean. Perform the beam erosion at each of these scales. Defaults to None.
            scales_as_fwhm (bool, optional): Use each scale as the FWHM of the Gaussian, as in earlier versions, rather than the Gaussian wsclean uses for it. Intended for comparisons. Defaults to False.

        Returns:
    MaskLike: The eroded beam shape. For no/single scale the dtype of ``mask`` is
        preserved; for multiple scales a float64 bitmask is returned, where the
        n'th scale is stored as the n'th bit.
    """

    if not _has_beam_parameters(fits_header=fits_header):
        logger.warning(
            "Beam parameters missing. Not performing the beam shape erosion. "
        )
        return mask

    scales = [0] if scales is None else scales
    assert isinstance(scales, (list, tuple)), f"{scales} should be a list or tuple"
    scales = _sort_scales(scales=scales)
    logger.info(f"Eroding across {scales=}")

    out_mask = np.zeros(mask.shape)

    for index, scale in enumerate(scales):
        erode_mask = create_multi_scale_erosion(
            mask=mask,
            fits_header=fits_header,
            scale=scale,
            minimum_response=minimum_response,
            scale_is_fwhm=scales_as_fwhm,
        )
        out_mask[erode_mask == 1.0] += 2**index

    if len(scales) == 1:
        return out_mask.astype(mask.dtype)

    return out_mask


def _reproject_mask_image(
    mosaic_mask_path: Path, header: fits.Header, shape: tuple[int, ...]
) -> NDArray[np.floating]:
    """Reproject a mask image onto the pixel grid of a header. Nearest neighbour
    interpolation is used so that mask values, including bitmapped scales, are
    preserved.

    Args:
        mosaic_mask_path (Path): The mask image to reproject
        header (fits.Header): Header with the WCS of the output pixel grid
        shape (tuple[int, ...]): The two-dimensional shape of the output pixel grid

    Returns:
        NDArray[np.floating]: The reprojected mask, which is zero outside of the mosaic footprint
    """
    logger.info(f"Will extract {shape} from {mosaic_mask_path}")
    with fits.open(mosaic_mask_path) as mosaic_mask:
        extract_image, _footprint = reproject_interp(
            input_data=(
                np.squeeze(mosaic_mask[0].data),
                WCS(mosaic_mask[0].header).celestial,
            ),
            output_projection=WCS(header).celestial,
            shape_out=shape,
            order=0,
        )

    # Pixels outside the mosaic footprint are NaN, which would otherwise cast
    # to a large non-zero integer and be treated as a cleanable region
    reprojected_mask: NDArray[np.floating] = np.nan_to_num(extract_image, nan=0.0)
    return reprojected_mask


def extract_beam_mask_from_mosaic(
    fits_beam_image_path: Path,
    fits_mosaic_mask_names: FITSMaskNames,
    overwrite: bool = True,
) -> FITSMaskNames:
    """Extract a region based on an existing FITS image from a masked FITS
    image. Here a masked FITS image is intended to be one created by
    ``create_snr_mask_from_fits``. Both a ``mask_fits`` and a per-scale
    ``scale_mask_fits`` are extracted should they be provided, with the
    bitmapped values of the latter preserved.

    A simple cutout (e.g. ``CutOut2D``) might not work as intended, as
    when creating the field image (the intended use case here) might be
    regridded onto a unique pixel grid that does not correspond to one
    that would be constructed by wsclean.

    Args:
        fits_beam_image_path (Path): The template image with a valid WCS. This region is used to extract the masked region
        fits_mosaic_mask_names (FITSMaskNames): The previously masked image
        overwrite (bool): Passed to `fits.writeto`, and will overwrite files should they exist. Defaults to True.

    Raises:
        ValueError: Raised when neither a ``mask_fits`` or ``scale_mask_fits`` is provided to extract from

    Returns:
        FITSMaskNames: Container with the extracted masks, which are named after ``fits_beam_image_path``. Masks that were not extracted are None.
    """

    if (
        fits_mosaic_mask_names.mask_fits is None
        and fits_mosaic_mask_names.scale_mask_fits is None
    ):
        msg = "Expected a mask_fits or scale_mask_fits FITS image to extract from"
        raise ValueError(msg)

    assert fits_mosaic_mask_names.mask_fits is not None, (
        "Expected a mask FITS image to extract from"
    )
    mask_names = create_fits_mask_names(fits_image=fits_beam_image_path)

    with fits.open(fits_beam_image_path) as beam_image:
        header = beam_image[0].header
        beam_image_shape = beam_image[0].data.shape[-2:]

    beam_mask_fits: Path | None = None
    if fits_mosaic_mask_names.mask_fits is not None:
        extract_image = _reproject_mask_image(
            mosaic_mask_path=fits_mosaic_mask_names.mask_fits,
            header=header,
            shape=beam_image_shape,
        )
        logger.info("Clipping extract mask results (interpolation errors)")
        fits.writeto(
            filename=mask_names.mask_fits,
            data=np.clip(extract_image, 0.0, 1.0).astype(np.int32),
            header=header,
            overwrite=overwrite,
        )
        beam_mask_fits = mask_names.mask_fits

    beam_scale_mask_fits: Path | None = None
    if fits_mosaic_mask_names.scale_mask_fits is not None:
        # No clipping, as the bitmapped value of each pixel encodes its scales
        extract_scale_image = _reproject_mask_image(
            mosaic_mask_path=fits_mosaic_mask_names.scale_mask_fits,
            header=header,
            shape=beam_image_shape,
        )
        fits.writeto(
            filename=mask_names.scale_mask_fits,
            data=extract_scale_image.astype(np.float32),
            header=header,
            overwrite=overwrite,
        )
        beam_scale_mask_fits = mask_names.scale_mask_fits

    return FITSMaskNames(mask_fits=beam_mask_fits, scale_mask_fits=beam_scale_mask_fits)


def _get_signal_image(
    image: NDArray[np.floating] | None = None,
    rms: NDArray[np.floating] | None = None,
    background: NDArray[np.floating] | None = None,
    signal: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Construct a signal image from inputs. The signal is defined as:

    >>> (image - background) / rms

    Should an input to ``signal`` be provided it will take precedence.

    Args:
        image (NDArray[np.floating] | None, optional): The base image. Defaults to None.
        rms (NDArray[np.floating] | None, optional): The noise estimate over the image. Defaults to None.
        background (NDArray[np.floating] | None, optional): The background across the image. Defaults to None.
        signal (NDArray[np.floating] | None, optional): A pre-calculate signal image. Defaults to None.

    Raises:
        ValueError: Raised when no signal is provided and there is not enough information to calculate one.

    Returns:
        NDArray[np.floating]: The output signal array
    """
    if all(item is None for item in (image, background, rms, signal)):
        msg = "No input maps have been provided. "
        raise ValueError(msg)

    if signal is not None:
        return signal

    assert isinstance(image, np.ndarray), f"{image=}, but should be an array"
    assert isinstance(rms, np.ndarray), f"{rms=}, but should be an array"
    if background is None:
        logger.info("No background supplied, assuming zeros. ")
        background = np.zeros_like(image)

    assert isinstance(background, np.ndarray)
    out_signal: NDArray[np.floating] = (image - background) / rms

    return out_signal


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


def _create_signal_from_rmsbkg(
    image: Path | NDArray[np.floating],
    rms: Path | NDArray[np.floating],
    bkg: Path | NDArray[np.floating],
) -> NDArray[np.floating]:
    """Form the signal image from the rms and background"""
    logger.info("Creating signal image")

    if isinstance(image, Path):
        with fits.open(image) as in_fits:
            logger.info(f"Loading {image}")
            image = in_fits[0].data

    assert isinstance(image, np.ndarray), (
        f"Expected the image to be a numpy array by now, instead have {type(image)}"
    )

    if isinstance(bkg, Path):
        with fits.open(bkg) as in_fits:
            logger.info(f"Loading {bkg=}")
            bkg = in_fits[0].data

    assert isinstance(bkg, np.ndarray), (
        f"Expected the background to be a numpy array by now, instead have {type(bkg)}"
    )
    logger.info("Subtracting background")
    image -= bkg

    if isinstance(rms, Path):
        with fits.open(rms) as in_fits:
            logger.info(f"Loading {rms=}")
            rms = in_fits[0].data

    assert isinstance(rms, np.ndarray), (
        f"Expected the rms to be a numpy array by now, instead have {type(rms)}"
    )
    logger.info("Dividing by rms")
    image /= rms

    return np.array(image)


def _need_to_make_signal(masking_options: MaskingOptions) -> bool:
    """Isolated functions to consider whether a signal image is needed. The minimum
    absolute clip only operates on the image as part of the flood fill. Otherwise the
    ``base_snr_clip`` is applied, which needs a signal image."""
    return not (masking_options.flood_fill and masking_options.flood_fill_use_mac)


# TODO: Need to rename this function
# TODO: This should be made into a dispatcher type function
def create_snr_mask_from_fits(
    fits_image_path: Path,
    masking_options: MaskingOptions,
    fits_rms_path: Path | None,
    fits_bkg_path: Path | None,
    create_signal_fits: bool = False,
    overwrite: bool = True,
) -> FITSMaskNames:
    """Create a mask for an input FITS image based on a signal to noise given a corresponding pair of RMS and background FITS images.

    Internally should a signal image be needed it is computed as something akin to:
    > signal = (image - background) / rms

    This is done in a staged manner to minimise the number of (potentially large) images
    held in memory.

    Each of the input images needs to share the same shape. This means that compression
    features offered by some tooling (e.g. BANE --compress) can not be used.

    Depending on the `MaskingOptions` used the signal image may not be needed.

    Without the flood fill, pixels above ``base_snr_clip`` form the mask.

    Args:
        fits_image_path (Path): Path to the FITS file containing an image
        masking_options (MaskingOptions): Configurable on the masking operation procedure.
        fits_rms_path (Path | None, optional): Path to the FITS file with an RMS image corresponding to ``fits_image_path``. Defaults to None.
        fits_bkg_path (Path | None, optional): Path to the FITS file with an background image corresponding to ``fits_image_path``. Defaults to None.
        create_signal_fits (bool, optional): Create an output signal map. Defaults to False.
        overwrite (bool): Passed to `fits.writeto`, and will overwrite files should they exist. Defaults to True.

    Raises:
        ValueError: Raised when a signal image is needed but the RMS or background FITS images are not provided

    Returns:
        FITSMaskNames: Container describing the FITS images that were written, with unwritten products set to None. ``signal_fits`` is only set if ``create_signal_fits`` is True and a signal image was needed. When ``beam_shape_erode`` is used with multiple scales (and the beam is known) the bit-packed mask is written to ``scale_mask_fits``, otherwise the mask is written to ``mask_fits``.
    """
    # Short cut to the convolve first
    if masking_options.convolve_first:
        logger.warning(
            "Using the convolve then erode algorithm, some options may be ignored"
        )
        return create_convolved_erosion_mask(
            fits_image_path=fits_image_path,
            masking_options=masking_options,
            overwrite=overwrite,
        )

    mask_names = create_fits_mask_names(
        fits_image=fits_image_path, include_signal_path=create_signal_fits
    )

    with fits.open(fits_image_path) as fits_image:
        fits_header = fits_image[0].header
        signal_data = fits_image[0].data

    signal_fits: Path | None = None
    if _need_to_make_signal(masking_options=masking_options):
        if fits_rms_path is None or fits_bkg_path is None:
            msg = "A signal image is needed, which requires paths to the RMS and background FITS images"
            if masking_options.flood_fill_use_mac and not masking_options.flood_fill:
                msg += (
                    ". Note that flood_fill_use_mac is only used when flood_fill=True"
                )
            raise ValueError(msg)

        signal_data = _create_signal_from_rmsbkg(
            image=signal_data, rms=fits_rms_path, bkg=fits_bkg_path
        )

        if create_signal_fits:
            logger.info(f"Writing {mask_names.signal_fits}")
            fits.writeto(
                filename=mask_names.signal_fits,
                data=signal_data,
                header=fits_header,
                overwrite=overwrite,
            )
            signal_fits = mask_names.signal_fits

    # Following the help in wsclean:
    # WSClean accepts masks in CASA format and in fits file format. A mask is a
    # normal, single polarization image file, where all zero values are interpreted
    # as being not masked, and all non-zero values are interpreted as masked. In the
    # case of a fits file, the file may either contain a single frequency or it may
    # contain a cube of images.
    if masking_options.flood_fill:
        # TODO: The image and signal masks both don't need to be inputs. Image is only used
        # if mac = True
        mask_data = create_flood_fill_mask(
            base_image=np.squeeze(signal_data),
            masking_options=masking_options,
        ).astype(np.float32)
        mask_data = mask_data.reshape(signal_data.shape)
    else:
        logger.info(f"Clipping using a {masking_options.base_snr_clip=}")
        mask_data = (signal_data > masking_options.base_snr_clip).astype(float)

    if masking_options.beam_shape_erode:
        mask_data = beam_shape_erode(
            mask=mask_data,
            fits_header=fits_header,
            minimum_response=masking_options.beam_shape_erode_minimum_response,
            scales=masking_options.beam_shape_erode_scales,
            scales_as_fwhm=masking_options.beam_shape_erode_scales_as_fwhm,
        )

    # Multiple erosion scales are bit-packed per pixel, so belong in the scale mask.
    # Without beam parameters no erosion happens and the mask is a plain 0/1 mask
    scales = masking_options.beam_shape_erode_scales
    is_scale_mask = (
        masking_options.beam_shape_erode
        and _has_beam_parameters(fits_header=fits_header)
        and (scales is not None and len(scales) > 1)
    )
    output_fits = mask_names.scale_mask_fits if is_scale_mask else mask_names.mask_fits

    logger.info(f"Writing {output_fits}")
    fits.writeto(
        filename=output_fits,
        data=mask_data,
        header=fits_header,
        overwrite=overwrite,
    )

    # Only report the files that were written
    if is_scale_mask:
        return FITSMaskNames(signal_fits=signal_fits, scale_mask_fits=output_fits)
    return FITSMaskNames(signal_fits=signal_fits, mask_fits=output_fits)


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
    image_data = grey_opening(input=image_data, size=scale, mode="wrap")

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


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Simple utility functions to create masks from FITS images. "
    )

    subparser = parser.add_subparsers(
        dest="mode", help="Operation mode of flint_bandpass"
    )

    fits_parser = subparser.add_parser(
        "mask",
        help="Create a mask for an image, potentially using its RMS and BKG images (e.g. outputs from BANE). Output FITS image will default to the image with a mask suffix.",
    )
    fits_parser.add_argument("image", type=Path, help="Path to the input image. ")
    fits_parser = add_options_to_parser(
        parser=fits_parser, options_class=MaskingOptions
    )

    fits_parser.add_argument(
        "--rms-fits", type=Path, help="Path to the RMS of the input image. "
    )
    fits_parser.add_argument(
        "--bkg-fits", type=Path, help="Path to the BKG of the input image. "
    )
    fits_parser.add_argument(
        "--save-signal",
        action="store_true",
        help="Save the signal image internally generated (should it be generated)",
    )

    extract_parser = subparser.add_parser(
        "extractmask",
        help="Extract a beam FITS masked region from a larger FITS mask mosaic image. ",
    )

    extract_parser.add_argument(
        "beam_image",
        type=Path,
        help="The FITS image with the WCS that will be used to extract the mask region from",
    )

    extract_parser.add_argument(
        "mosaic_image",
        type=Path,
        help="The field mosaic image that will be used as a basis to extract a region mask from. Should it end with .scalemask.fits it is treated as a per-scale mask, and its bitmapped values are preserved",
    )

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "mask":
        masking_options = create_options_from_parser(
            parser_namespace=args, options_class=MaskingOptions
        )
        # Dispatches to the convolve first approach if requested
        create_snr_mask_from_fits(
            fits_image_path=args.image,
            fits_rms_path=args.rms_fits,
            fits_bkg_path=args.bkg_fits,
            create_signal_fits=args.save_signal,
            masking_options=masking_options,
        )

    elif args.mode == "extractmask":
        mosaic_mask_names = (
            FITSMaskNames(scale_mask_fits=args.mosaic_image)
            if args.mosaic_image.name.endswith(".scalemask.fits")
            else FITSMaskNames(mask_fits=args.mosaic_image)
        )
        extract_beam_mask_from_mosaic(
            fits_beam_image_path=args.beam_image,
            fits_mosaic_mask_names=mosaic_mask_names,
        )
    else:
        logger.error(f"Supplied mode {args.mode} is not known. ")
        parser.print_help()


if __name__ == "__main__":
    cli()
