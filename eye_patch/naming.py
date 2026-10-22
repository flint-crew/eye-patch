from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from eye_patch.logging import logger

# The suffixes that name each product after its image, e.g. image.mask.fits
MASK_SUFFIX = ".mask.fits"
SCALE_MASK_SUFFIX = ".scalemask.fits"
SIGNAL_SUFFIX = ".signal.fits"


class FITSMaskNames(NamedTuple):
    """Contains the names of the FITS images created when creating a mask image.
    When returned by ``create_fits_mask_names`` these are only the candidate names,
    and do not mean that they are necessarily created. When returned by a masking
    function only the products that were written are set, and others are None.
    """

    mask_fits: Path | None = None
    """Name of the mask FITS file"""
    signal_fits: Path | None = None
    """Name of the signal FITS file"""
    scale_mask_fits: Path | None = None
    """Path to a FITS file that describes per-scale clean masks. Scales are represented with bit-mapped values."""


def create_fits_mask_names(
    fits_image: str | Path, include_signal_path: bool = False
) -> FITSMaskNames:
    """Create the names that will be used when generate FITS mask products

    Args:
        fits_image (str | Path): Base name of the output files
        include_signal_path (bool, optional): If True, also include ``signal_fits`` in the output. Defaults to False.

    Returns:
        FITSMaskNames: collection of names used for the signal and mask FITS images
    """
    fits_image = Path(fits_image)

    fits_signal = fits_image.with_suffix(SIGNAL_SUFFIX) if include_signal_path else None
    fits_mask = fits_image.with_suffix(MASK_SUFFIX)
    fits_scale_mask = fits_image.with_suffix(SCALE_MASK_SUFFIX)

    return FITSMaskNames(
        signal_fits=fits_signal, mask_fits=fits_mask, scale_mask_fits=fits_scale_mask
    )


def mask_fits_from_scale_mask(scale_mask_fits: str | Path) -> Path:
    """The name of the mask that corresponds to a per-scale mask. Following
    ``create_fits_mask_names``, ``image.scalemask.fits`` becomes ``image.mask.fits``.
    Any other name has its suffix replaced, so ``image.fits`` becomes
    ``image.mask.fits``.

    Args:
        scale_mask_fits (str | Path): Path to the per-scale mask

    Returns:
        Path: Path to the corresponding mask, in the same directory
    """
    scale_mask_fits = Path(scale_mask_fits)

    # with_suffix only replaces the last suffix, which would keep the .scalemask
    if scale_mask_fits.name.endswith(SCALE_MASK_SUFFIX):
        return scale_mask_fits.with_name(
            scale_mask_fits.name.removesuffix(SCALE_MASK_SUFFIX) + MASK_SUFFIX
        )
    return scale_mask_fits.with_suffix(MASK_SUFFIX)


def check_output_fits(
    output_fits: str | Path,
    overwrite: bool,
    input_fits: Iterable[str | Path | None] = (),
) -> None:
    """Check that an output FITS image may be written, before any work is done to
    make it. An existing output is an error unless ``overwrite`` is set, in which
    case it is replaced with a warning. An output that is also one of the inputs,
    even when reached through another path such as a symbolic link, is reported as
    such, as replacing it would lose that input.

    Args:
        output_fits (str | Path): The output that will be written
        overwrite (bool): Whether an existing output may be replaced
        input_fits (Iterable[str | Path | None], optional): The inputs of the operation, where None is skipped. Defaults to ().

    Raises:
        FileExistsError: Raised when the output already exists and ``overwrite`` is False
    """
    output_fits = Path(output_fits)
    if not output_fits.exists():
        return

    is_input = any(
        Path(path).exists() and output_fits.samefile(path)
        for path in input_fits
        if path is not None
    )
    if not overwrite:
        msg = (
            f"{output_fits} is also an input, which would be replaced"
            if is_input
            else f"{output_fits} already exists"
        )
        msg += ". Set overwrite=True (--overwrite on the CLI) to replace it"
        raise FileExistsError(msg)

    if is_input:
        logger.warning(f"Replacing the input {output_fits} with an output")
    else:
        logger.warning(f"Overwriting the existing {output_fits}")
