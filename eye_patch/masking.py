"""Create clean masks from FITS images. This is the ``mask`` sub-command of the
``eye-patch`` CLI."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from astropy.io import fits
from capn_crunch import add_options_to_parser, create_options_from_parser

from eye_patch.convolve import create_convolved_erosion_mask
from eye_patch.erosion import beam_shape_erode
from eye_patch.flood_fill import create_flood_fill_mask
from eye_patch.logging import logger
from eye_patch.naming import FITSMaskNames, create_fits_mask_names
from eye_patch.options import MaskingOptions
from eye_patch.pixel_utils import _has_beam_parameters
from eye_patch.snr import _create_signal_from_rmsbkg


def _need_to_make_signal(masking_options: MaskingOptions) -> bool:
    """Isolated functions to consider whether a signal image is needed. The minimum
    absolute clip only operates on the image as part of the flood fill. Otherwise the
    ``base_snr_clip`` is applied, which needs a signal image."""
    return not (masking_options.flood_fill and masking_options.flood_fill_use_mac)


def create_native_resolution_mask(
    fits_image_path: Path,
    masking_options: MaskingOptions,
    fits_rms_path: Path | None,
    fits_bkg_path: Path | None,
    create_signal_fits: bool = False,
    overwrite: bool = True,
) -> FITSMaskNames:
    """Create a mask for an input FITS image based on a signal to noise given a corresponding pair of RMS and background FITS images.

    The image is masked at its native resolution, which is the default approach of
    ``create_mask_from_fits``.

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


def create_mask_from_fits(
    fits_image_path: Path,
    masking_options: MaskingOptions,
    fits_rms_path: Path | None,
    fits_bkg_path: Path | None,
    create_signal_fits: bool = False,
    overwrite: bool = True,
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
        overwrite (bool): Passed to `fits.writeto`, and will overwrite files should they exist. Defaults to True.

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
    )
