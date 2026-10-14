from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest
from astropy.io import fits

from eye_patch.convolve import convolve_image_by_scale
from eye_patch.masking import create_snr_mask_from_fits
from eye_patch.naming import FITSMaskNames
from eye_patch.options import MaskingOptions

if TYPE_CHECKING:
    from pathlib import Path

SHAPE = (100, 100)


def test_fits_masking_convolve_first(fits_dir):
    """A basic check to see if the convolve first algorithm is working, where
    working be no errors mate"""
    masking_options = MaskingOptions(
        flood_fill_use_mac=True,
        flood_fill_positive_seed_clip=0.5,
        convolve_first=True,
    )
    names = create_snr_mask_from_fits(
        fits_image_path=fits_dir / "image.fits",
        fits_bkg_path=None,
        fits_rms_path=None,
        masking_options=masking_options,
    )

    assert isinstance(names, FITSMaskNames)
    assert names.scale_mask_fits is not None
    assert names.scale_mask_fits.exists()
    assert names.signal_fits is None
    assert names.mask_fits is None

    mask_data = fits.getdata(names.scale_mask_fits)
    assert mask_data.dtype.kind == "f", "A bool mask cannot be written to FITS"
    valid = np.sum(mask_data)
    assert valid == np.prod(SHAPE)


def _disk_centroid(image: np.ndarray) -> tuple[float, float]:
    """Intensity weighted centroid of the positive pixels in an image"""
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
    weights = np.clip(image, 0.0, None)
    return (
        float(np.sum(yy * weights) / np.sum(weights)),
        float(np.sum(xx * weights) / np.sum(weights)),
    )


@pytest.mark.parametrize("scale", [1, 2, 3, 4, 8, 16])
def test_convolve_image_by_scale_is_not_shifted(scale: int) -> None:
    """Small scales used to crash, and even sized opening filters and kernels
    shifted the convolved image (and so the mask) away from the source"""
    yy, xx = np.mgrid[:128, :128]
    image = (((xx - 64) ** 2 + (yy - 64) ** 2) <= 15**2).astype(np.float32)

    convolved = convolve_image_by_scale(image_data=image, scale=scale)

    assert convolved.shape == image.shape
    assert convolved.dtype == image.dtype
    assert np.allclose(_disk_centroid(image=convolved), (64.0, 64.0), atol=0.1)


def test_fits_masking_convolve_first_sorts_scales(
    fits_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The convolve first approach also assigns bits in ascending scale order"""
    masking_options = MaskingOptions(
        flood_fill_use_mac=True,
        flood_fill_positive_seed_clip=0.5,
        convolve_first=True,
        beam_shape_erode_scales=(4, 0),
    )
    with caplog.at_level(logging.WARNING, logger="eye-patch"):
        create_snr_mask_from_fits(
            fits_image_path=fits_dir / "image.fits",
            fits_bkg_path=None,
            fits_rms_path=None,
            masking_options=masking_options,
        )
    assert any("ascending order" in record.message for record in caplog.records)


def test_fits_masking_convolve_first_with_nan(fits_dir: Path) -> None:
    """A single NaN pixel used to make the entire convolved image NaN, leaving
    that scale with an empty mask"""
    image_path = fits_dir / "image.fits"
    image = np.ones(SHAPE)
    image[50, 50] = np.nan
    fits.writeto(image_path, image, overwrite=True)

    masking_options = MaskingOptions(
        flood_fill_use_mac=True,
        flood_fill_positive_seed_clip=0.5,
        convolve_first=True,
        beam_shape_erode_scales=(0, 4),
    )
    names = create_snr_mask_from_fits(
        fits_image_path=image_path,
        fits_bkg_path=None,
        fits_rms_path=None,
        masking_options=masking_options,
    )

    assert names.scale_mask_fits is not None
    mask_data = fits.getdata(names.scale_mask_fits)
    assert np.all(np.isfinite(mask_data))
    # The second scale is the second bit
    scale_four = (mask_data.astype(int) & 2) > 0
    assert np.mean(scale_four) > 0.9
