"""Erode masks with the shape of the restoring beam, optionally across the scales
of a wsclean multi-scale clean"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from radio_beam import Beam
from scipy.ndimage import (
    binary_erosion as scipy_binary_erosion,  # Rename to distinguish from skimage
)
from scipy.signal import fftconvolve

from eye_patch.logging import logger
from eye_patch.pixel_utils import has_beam_parameters

if TYPE_CHECKING:
    from astropy.io import fits

# Masks that may be written to a FITS file. Booleans are deliberately excluded, as
# fits.writeto raises on a boolean array, so type checking stops one reaching it.
MaskLike: TypeAlias = NDArray[np.floating]
# Masks that are accepted as either booleans or floats, e.g. by the erosion helpers
FloatOrBoolMask: TypeAlias = NDArray[np.floating] | NDArray[np.bool]


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
    if not has_beam_parameters(fits_header=fits_header):
        logger.warning(
            "Beam parameters missing. Not performing the beam shape erosion. "
        )
        return mask.astype(bool, copy=False)

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

    if not has_beam_parameters(fits_header=fits_header):
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
