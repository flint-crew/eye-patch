"""Operations on per-scale masks, where the n'th scale of each pixel is stored as
its n'th bit. This is the ``scales`` sub-command of the ``eye-patch`` CLI."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits

from eye_patch.logging import logger
from eye_patch.naming import check_output_fits, mask_fits_from_scale_mask

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Header keywords that describe integer data, which a float image cannot carry
INTEGER_ONLY_KEYWORDS = ("BLANK", "BSCALE", "BZERO")


def flatten_scale_mask(scale_mask: NDArray[np.floating]) -> NDArray[np.float32]:
    """Flatten a per-scale mask into a single mask, where a pixel that is set at
    any scale becomes 1 and every other pixel 0. Non-finite pixels are treated as
    not masked.

    Args:
        scale_mask (NDArray[np.floating]): The per-scale mask, with its scales as bitmapped values

    Returns:
        NDArray[np.float32]: The flattened mask, with the same shape as ``scale_mask``
    """
    # Built in place, to avoid a second full-size temporary for large mosaics
    masked = scale_mask != 0
    masked &= np.isfinite(scale_mask)
    flattened: NDArray[np.float32] = masked.astype(np.float32)
    return flattened


def flatten_scale_mask_fits(
    scale_mask_fits: str | Path,
    output_fits: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Flatten a per-scale mask FITS image into a single mask (see
    ``flatten_scale_mask``), written with the header of the per-scale mask.

    An existing output is only replaced with ``overwrite``, which is checked before
    anything is read (see ``check_output_fits``). This includes an output that is
    ``scale_mask_fits`` itself, whose scales would be lost.

    Args:
        scale_mask_fits (str | Path): The per-scale mask to flatten
        output_fits (str | Path | None, optional): Where to write the flattened mask. If None it is named after ``scale_mask_fits`` (see ``mask_fits_from_scale_mask``), so ``image.scalemask.fits`` becomes ``image.mask.fits``. Defaults to None.
        overwrite (bool, optional): Replace the output should it already exist, with a warning. Defaults to False.

    Raises:
        FileExistsError: Raised when the output already exists and ``overwrite`` is False

    Returns:
        Path: The flattened mask that was written
    """
    scale_mask_fits = Path(scale_mask_fits)
    output_fits = (
        mask_fits_from_scale_mask(scale_mask_fits=scale_mask_fits)
        if output_fits is None
        else Path(output_fits)
    )

    check_output_fits(
        output_fits=output_fits, overwrite=overwrite, input_fits=(scale_mask_fits,)
    )

    with fits.open(scale_mask_fits) as scale_mask:
        header = scale_mask[0].header.copy()
        mask = flatten_scale_mask(scale_mask=scale_mask[0].data)

    # The flattened mask is written as floats, so keywords that only describe integer
    # data (e.g. from an integer per-scale mask) would make its header invalid
    for keyword in INTEGER_ONLY_KEYWORDS:
        header.remove(keyword, ignore_missing=True)

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
    flatten_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output should it already exist, including the per-scale mask itself. Without it an existing output is an error",
    )

    return parser


def cli(args: Namespace | None = None) -> None:
    """Run the ``scales`` sub-command

    Args:
        args (Namespace | None, optional): Arguments parsed by ``get_parser``. If None they are parsed from the command line. Defaults to None.

    Raises:
        ValueError: Raised for an operation that has no branch here, e.g. one added to ``get_parser`` alone
    """
    args = get_parser().parse_args() if args is None else args

    if args.operation == "flatten":
        flatten_scale_mask_fits(
            scale_mask_fits=args.scale_mask,
            output_fits=args.output_fits,
            overwrite=args.overwrite,
        )
    else:
        msg = f"The scales operation {args.operation!r} is not known"
        raise ValueError(msg)
