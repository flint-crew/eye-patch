"""Create clean masks from FITS images. This is the ``mask`` sub-command of the
``eye-patch`` CLI."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits
from capn_crunch import add_options_to_parser, create_options_from_parser

from eye_patch.convolve import create_convolved_erosion_mask
from eye_patch.erosion import beam_shape_erode
from eye_patch.flood_fill import (
    _select_large_islands,
    create_flood_fill_mask,
    grow_low_snr_mask,
)
from eye_patch.logging import logger
from eye_patch.mac import minimum_absolute_clip_threshold
from eye_patch.naming import (
    FITSMaskNames,
    check_output_fits,
    create_fits_mask_names,
)
from eye_patch.options import MaskingOptions
from eye_patch.pixel_utils import has_beam_parameters
from eye_patch.snr import create_signal_from_rmsbkg

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _need_to_make_signal(masking_options: MaskingOptions) -> bool:
    """Isolated functions to consider whether a signal image is needed. The minimum
    absolute clip operates on the image itself, with or without the flood fill,
    whereas the signal-to-noise clips need a signal image."""
    return not masking_options.use_mac


def _create_island_mask(
    base_image: NDArray[np.floating], masking_options: MaskingOptions
) -> NDArray[np.bool]:
    """Clip the base image into the islands of a clean mask, before any beam erosion.

    Islands are flood filled from bright seeds should ``flood_fill`` be set, and are
    otherwise the pixels above ``base_clip``. Should ``grow_low_snr_island`` be set,
    large islands of faint and diffuse emission are added in either case. With
    ``use_mac`` each clip is an increase factor of the minimum absolute clip, rather
    than a signal-to-noise level.

    Args:
        base_image (NDArray[np.floating]): The two-dimensional signal image, or the image itself should ``use_mac`` be set
        masking_options (MaskingOptions): Options to carry out masking

    Returns:
        NDArray[np.bool]: Mask of the pixels to clean
    """
    mac_threshold: NDArray[np.floating] | None = None
    if masking_options.use_mac:
        # The base, seed, flood and low SNR island clips share the same clipping level
        mac_threshold = minimum_absolute_clip_threshold(
            image=base_image,
            box_size=masking_options.mac_box_size,
            adaptive_max_depth=masking_options.mac_adaptive_max_depth,
            adaptive_box_step=masking_options.mac_adaptive_step_factor,
            adaptive_skew_delta=masking_options.mac_adaptive_skew_delta,
        )

    island_mask: NDArray[np.bool]
    if masking_options.flood_fill:
        island_mask = create_flood_fill_mask(
            base_image=base_image,
            masking_options=masking_options,
            mac_threshold=mac_threshold,
        )
    elif mac_threshold is not None:
        logger.info(f"Minimum absolute clip using a {masking_options.base_clip=}")
        island_mask = base_image > (mac_threshold * masking_options.base_clip)
    else:
        logger.info(f"Clipping using a {masking_options.base_clip=}")
        island_mask = base_image > masking_options.base_clip

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
                minimum_island_size=masking_options.grow_low_snr_island_min_pixels,
            )
        else:
            low_snr_mask = grow_low_snr_mask(
                signal=base_image,
                grow_low_snr=masking_options.grow_low_snr_island_clip,
                grow_low_island_size=masking_options.grow_low_snr_island_min_pixels,
            )
        island_mask[low_snr_mask] = True

    return island_mask


def create_native_resolution_mask(
    fits_image_path: Path,
    masking_options: MaskingOptions,
    fits_rms_path: Path | None,
    fits_bkg_path: Path | None,
    create_signal_fits: bool = False,
    overwrite: bool = False,
) -> FITSMaskNames:
    """Create a mask for an input FITS image based on a signal to noise given a corresponding pair of RMS and background FITS images, or on the minimum absolute clip of the image itself should ``use_mac`` be set.

    The image is masked at its native resolution, which is the default approach of
    ``create_mask_from_fits``.

    Internally should a signal image be needed it is computed as something akin to:
    > signal = (image - background) / rms

    This is done in a staged manner to minimise the number of (potentially large) images
    held in memory.

    Each of the input images needs to share the same shape. This means that compression
    features offered by some tooling (e.g. BANE --compress) can not be used.

    The minimum absolute clip operates on the image itself, so with ``use_mac`` the
    signal image is not needed.

    Islands are flood filled should ``flood_fill`` be set, and are otherwise the pixels
    above ``base_clip``. Large islands of faint emission are added in either case should
    ``grow_low_snr_island`` be set (see ``_create_island_mask``).

    Args:
        fits_image_path (Path): Path to the FITS file containing an image
        masking_options (MaskingOptions): Configurable on the masking operation procedure.
        fits_rms_path (Path | None, optional): Path to the FITS file with an RMS image corresponding to ``fits_image_path``. Defaults to None.
        fits_bkg_path (Path | None, optional): Path to the FITS file with an background image corresponding to ``fits_image_path``. Defaults to None.
        create_signal_fits (bool, optional): Create an output signal map. Defaults to False.
        overwrite (bool, optional): Replace the outputs should they already exist, with a warning. They are checked before the mask is made (see ``check_output_fits``). Defaults to False.

    Raises:
        ValueError: Raised when a signal image is needed but the RMS or background FITS images are not provided
        FileExistsError: Raised when an output already exists and ``overwrite`` is False

    Returns:
        FITSMaskNames: Container describing the FITS images that were written, with unwritten products set to None. ``signal_fits`` is only set if ``create_signal_fits`` is True and a signal image was needed. When ``beam_shape_erode`` is used with multiple scales (and the beam is known) the bit-packed mask is written to ``scale_mask_fits``, otherwise the mask is written to ``mask_fits``.
    """
    mask_names = create_fits_mask_names(
        fits_image=fits_image_path, include_signal_path=create_signal_fits
    )

    with fits.open(fits_image_path) as fits_image:
        fits_header = fits_image[0].header
        signal_data = fits_image[0].data

    # Multiple erosion scales are bit-packed per pixel, so belong in the scale mask.
    # Without beam parameters no erosion happens and the mask is a plain 0/1 mask
    scales = masking_options.multiscale_scales
    is_scale_mask = (
        masking_options.beam_shape_erode
        and has_beam_parameters(fits_header=fits_header)
        and (scales is not None and len(scales) > 1)
    )
    output_fits = mask_names.scale_mask_fits if is_scale_mask else mask_names.mask_fits
    need_signal = _need_to_make_signal(masking_options=masking_options)

    # The outputs are checked before the mask is made, so none is written by halves
    input_fits = (fits_image_path, fits_rms_path, fits_bkg_path)
    assert output_fits is not None
    check_output_fits(
        output_fits=output_fits, overwrite=overwrite, input_fits=input_fits
    )
    if need_signal and create_signal_fits:
        assert mask_names.signal_fits is not None
        check_output_fits(
            output_fits=mask_names.signal_fits,
            overwrite=overwrite,
            input_fits=input_fits,
        )

    signal_fits: Path | None = None
    if need_signal:
        if fits_rms_path is None or fits_bkg_path is None:
            msg = "A signal image is needed, which requires paths to the RMS and background FITS images"
            raise ValueError(msg)

        signal_data = create_signal_from_rmsbkg(
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
    mask_data = _create_island_mask(
        base_image=np.squeeze(signal_data), masking_options=masking_options
    ).astype(np.float32)
    mask_data = mask_data.reshape(signal_data.shape)

    if masking_options.beam_shape_erode:
        mask_data = beam_shape_erode(
            mask=mask_data,
            fits_header=fits_header,
            minimum_response=masking_options.beam_shape_erode_minimum_response,
            scales=masking_options.multiscale_scales,
            scales_as_fwhm=masking_options.beam_shape_erode_scales_as_fwhm,
        )

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


def create_mask_from_fits(
    fits_image_path: Path,
    masking_options: MaskingOptions,
    fits_rms_path: Path | None = None,
    fits_bkg_path: Path | None = None,
    create_signal_fits: bool = False,
    overwrite: bool = False,
) -> FITSMaskNames:
    """Create a mask for an input FITS image, dispatching to the approach selected by
    the ``masking_options``. Should ``convolve_first`` be set the image is first
    convolved to each scale (see ``create_convolved_erosion_mask``), which ignores
    ``fits_rms_path``, ``fits_bkg_path`` and ``create_signal_fits``. Otherwise the
    image is masked at its native resolution (see ``create_native_resolution_mask``).

    Args:
        fits_image_path (Path): Path to the FITS file containing an image
        masking_options (MaskingOptions): Configurable on the masking operation procedure.
        fits_rms_path (Path | None, optional): Path to the FITS file with an RMS image corresponding to ``fits_image_path``. Defaults to None.
        fits_bkg_path (Path | None, optional): Path to the FITS file with an background image corresponding to ``fits_image_path``. Defaults to None.
        create_signal_fits (bool, optional): Create an output signal map. Defaults to False.
        overwrite (bool, optional): Replace the outputs should they already exist, with a warning. Without it an existing output raises a FileExistsError. Defaults to False.

    Returns:
        FITSMaskNames: Container describing the FITS images that were written by the selected approach, with unwritten products set to None
    """
    if masking_options.convolve_first:
        logger.warning(
            "Using the convolve then erode algorithm, some options may be ignored"
        )
        return create_convolved_erosion_mask(
            fits_image_path=fits_image_path,
            masking_options=masking_options,
            overwrite=overwrite,
        )

    return create_native_resolution_mask(
        fits_image_path=fits_image_path,
        masking_options=masking_options,
        fits_rms_path=fits_rms_path,
        fits_bkg_path=fits_bkg_path,
        create_signal_fits=create_signal_fits,
        overwrite=overwrite,
    )


def get_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    """Add the arguments of the ``mask`` sub-command to a parser

    Args:
        parser (ArgumentParser | None, optional): The parser to add the arguments to, e.g. a sub-parser of the ``eye-patch`` CLI. If None a new parser is created. Defaults to None.

    Returns:
        ArgumentParser: The parser with the ``mask`` arguments added
    """
    if parser is None:
        parser = ArgumentParser(description=__doc__)

    parser.add_argument("image", type=Path, help="Path to the input image. ")
    parser = add_options_to_parser(parser=parser, options_class=MaskingOptions)

    parser.add_argument(
        "--rms-fits", type=Path, help="Path to the RMS of the input image. "
    )
    parser.add_argument(
        "--bkg-fits", type=Path, help="Path to the BKG of the input image. "
    )
    parser.add_argument(
        "--save-signal",
        action="store_true",
        help="Save the signal image internally generated (should it be generated)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the outputs should they already exist. Without it an existing output is an error",
    )

    return parser


def cli(args: Namespace | None = None) -> None:
    """Run the ``mask`` sub-command

    Args:
        args (Namespace | None, optional): Arguments parsed by ``get_parser``. If None they are parsed from the command line. Defaults to None.
    """
    args = get_parser().parse_args() if args is None else args

    masking_options = create_options_from_parser(
        parser_namespace=args, options_class=MaskingOptions
    )
    # Dispatches to the convolve first approach if requested
    create_mask_from_fits(
        fits_image_path=args.image,
        fits_rms_path=args.rms_fits,
        fits_bkg_path=args.bkg_fits,
        create_signal_fits=args.save_signal,
        masking_options=masking_options,
        overwrite=args.overwrite,
    )
