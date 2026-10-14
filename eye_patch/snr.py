"""Construct the signal image, ``(image - background) / rms``, used to clip on
signal-to-noise"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits

from eye_patch.logging import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


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


def _create_signal_from_rmsbkg(
    image: Path | NDArray[np.floating],
    rms: Path | NDArray[np.floating],
    bkg: Path | NDArray[np.floating],
) -> NDArray[np.floating]:
    """Generate a signal image from inputs. The final signal image is formed
    using inplace operations to attempt to avoid duplication, which can become a
    problem for large input images. Be aware of this should you pass arrays in
    as inputs.

    Args:
        image (Path | NDArray[np.floating]): The imafe. If a fits file the data from the first extension are loaded.
        rms (Path | NDArray[np.floating]): The background. If a files file the data from the first extension are loaded.
        bkg (Path | NDArray[np.floating]): The rms. if a fits file the data from the first extension are loaded.

    Returns:
        NDArray[np.floating]: A signal image
    """

    # NOTE: This is similar to ``_get_signal_image`` but intend to work
    # against larger images, performing operations inplace to avoid duplication.
    # Really only intended for very large images (e.g. many GBs in size)
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
