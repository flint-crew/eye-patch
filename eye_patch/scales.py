"""Operations on per-scale masks, where the n'th scale of each pixel is stored as
its n'th bit. This is the ``scales`` sub-command of the ``eye-patch`` CLI."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits

from eye_patch.logging import logger
from eye_patch.naming import mask_fits_from_scale_mask

if TYPE_CHECKING:
    from numpy.typing import NDArray


def flatten_scale_mask(scale_mask: NDArray[np.floating]) -> NDArray[np.float32]:
    """Flatten a per-scale mask into a single mask, where a pixel that is set at
    any scale becomes 1 and every other pixel 0. Non-finite pixels are treated as
    not masked.

    Args:
        scale_mask (NDArray[np.floating]): The per-scale mask, with its scales as bitmapped values

    Returns:
        NDArray[np.float32]: The flattened mask, with the same shape as ``scale_mask``
    """
    masked = np.isfinite(scale_mask) & (scale_mask != 0)
    flattened: NDArray[np.float32] = masked.astype(np.float32)
    return flattened


def flatten_scale_mask_fits(
    scale_mask_fits: Path, output_fits: Path | None = None, overwrite: bool = True
) -> Path:
    """Flatten a per-scale mask FITS image into a single mask (see
    ``flatten_scale_mask``), written with the header of the per-scale mask.

    Args:
        scale_mask_fits (Path): The per-scale mask to flatten
        output_fits (Path | None, optional): Where to write the flattened mask. If None it is named after ``scale_mask_fits`` (see ``mask_fits_from_scale_mask``), so ``image.scalemask.fits`` becomes ``image.mask.fits``. Defaults to None.
        overwrite (bool, optional): Passed to `fits.writeto`, and will overwrite files should they exist. Defaults to True.

    Returns:
        Path: The flattened mask that was written
    """
    output_fits = (
        mask_fits_from_scale_mask(scale_mask_fits=scale_mask_fits)
        if output_fits is None
        else output_fits
    )

    with fits.open(scale_mask_fits) as scale_mask:
        header = scale_mask[0].header
        mask = flatten_scale_mask(scale_mask=scale_mask[0].data)

    logger.info(f"Writing the flattened {scale_mask_fits} to {output_fits}")
    fits.writeto(filename=output_fits, data=mask, header=header, overwrite=overwrite)
    return output_fits


def get_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    """Add the operations of the ``scales`` sub-command to a parser

    Args:
        parser (ArgumentParser | None, optional): The parser to add the operations to, e.g. a sub-parser of the ``eye-patch`` CLI. If None a new parser is created. Defaults to None.

    Returns:
        ArgumentParser: The parser with the ``scales`` operations added
    """
    if parser is None:
        parser = ArgumentParser(description=__doc__)

    operations = parser.add_subparsers(
        dest="operation", required=True, help="Operation on a per-scale mask"
    )

    flatten_parser = operations.add_parser(
        "flatten",
        help="Flatten a per-scale mask into a mask of the pixels that are set at any scale",
    )
    flatten_parser.add_argument(
        "scale_mask", type=Path, help="Path to the per-scale mask to flatten"
    )
    flatten_parser.add_argument(
        "--output-fits",
        type=Path,
        default=None,
        help="Path to write the flattened mask to. Defaults to the per-scale mask with its .scalemask.fits (or .fits) suffix replaced by .mask.fits",
    )

    return parser


def cli(args: Namespace | None = None) -> None:
    """Run the ``scales`` sub-command

    Args:
        args (Namespace | None, optional): Arguments parsed by ``get_parser``. If None they are parsed from the command line. Defaults to None.
    """
    args = get_parser().parse_args() if args is None else args

    if args.operation == "flatten":
        flatten_scale_mask_fits(
            scale_mask_fits=args.scale_mask, output_fits=args.output_fits
        )
