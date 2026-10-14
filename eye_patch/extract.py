"""Extract the region of a single beam from a larger mask, e.g. one made from a
field mosaic. This is the ``extractmask`` sub-command of the ``eye-patch`` CLI."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

from eye_patch.logging import logger
from eye_patch.naming import FITSMaskNames, create_fits_mask_names

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    ``create_mask_from_fits``. Both a ``mask_fits`` and a per-scale
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


def get_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    """Add the arguments of the ``extractmask`` sub-command to a parser

    Args:
        parser (ArgumentParser | None, optional): The parser to add the arguments to, e.g. a sub-parser of the ``eye-patch`` CLI. If None a new parser is created. Defaults to None.

    Returns:
        ArgumentParser: The parser with the ``extractmask`` arguments added
    """
    if parser is None:
        parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        "beam_image",
        type=Path,
        help="The FITS image with the WCS that will be used to extract the mask region from",
    )

    parser.add_argument(
        "mosaic_image",
        type=Path,
        help="The field mosaic image that will be used as a basis to extract a region mask from. Should it end with .scalemask.fits it is treated as a per-scale mask, and its bitmapped values are preserved",
    )

    return parser


def cli(args: Namespace | None = None) -> None:
    """Run the ``extractmask`` sub-command

    Args:
        args (Namespace | None, optional): Arguments parsed by ``get_parser``. If None they are parsed from the command line. Defaults to None.
    """
    args = get_parser().parse_args() if args is None else args

    mosaic_mask_names = (
        FITSMaskNames(scale_mask_fits=args.mosaic_image)
        if args.mosaic_image.name.endswith(".scalemask.fits")
        else FITSMaskNames(mask_fits=args.mosaic_image)
    )
    extract_beam_mask_from_mosaic(
        fits_beam_image_path=args.beam_image,
        fits_mosaic_mask_names=mosaic_mask_names,
    )
