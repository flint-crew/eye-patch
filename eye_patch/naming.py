from pathlib import Path
from typing import NamedTuple


class FITSMaskNames(NamedTuple):
    """Contains the names of the FITS images created when creating a mask image/
    These are only the names, and do not mean that they are necessarily created.
    """

    mask_fits: Path
    """Name of the mask FITS file"""
    signal_fits: Path | None = None
    """Name of the signal FITS file"""
    scale_mask_fits: Path | None = None
    """Path to a FITS file that describes per-scale clean masks. Scales are represented with a bit-mapped values,"""


def create_fits_mask_names(
    fits_image: str | Path, include_signal_path: bool = False
) -> FITSMaskNames:
    """Create the names that will be used when generate FITS mask products

    Args:
        fits_image (Union[str,]Path): Base name of the output files
        include_signal_path (bool, optional): If True, also include ``signal_fits`` in the output. Defaults to False.

    Returns:
        FITSMaskNames: collection of names used for the signal and mask FITS images
    """
    fits_image = Path(fits_image)

    fits_signal = (
        fits_image.with_suffix(".signal.fits") if include_signal_path else None
    )
    fits_mask = fits_image.with_suffix(".mask.fits")
    fits_scale_mask = fits_image.with_suffix(".scalemask.fits")

    return FITSMaskNames(
        signal_fits=fits_signal, mask_fits=fits_mask, scale_mask_fits=fits_scale_mask
    )
