from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING

import numpy as np
import pytest
from astropy.io import fits

from eye_patch import convolve
from eye_patch.convolve import convolve_image_by_scale
from eye_patch.masking import create_mask_from_fits
from eye_patch.naming import FITSMaskNames
from eye_patch.options import MaskingOptions

if TYPE_CHECKING:
    from pathlib import Path

SHAPE = (100, 100)


def test_fits_masking_convolve_first(fits_dir):
    """A basic check to see if the convolve first algorithm is working, where
    working be no errors mate"""
    masking_options = MaskingOptions(
        use_mac=True,
        seed_clip=0.5,
        convolve_first=True,
    )
    names = create_mask_from_fits(
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
        use_mac=True,
        seed_clip=0.5,
        convolve_first=True,
        multiscale_scales=(4, 0),
    )
    with caplog.at_level(logging.WARNING, logger="eye-patch"):
        create_mask_from_fits(
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
        use_mac=True,
        seed_clip=0.5,
        convolve_first=True,
        multiscale_scales=(0, 4),
    )
    names = create_mask_from_fits(
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


def test_fits_masking_convolve_first_will_not_replace_an_existing_mask(
    fits_dir: Path,
) -> None:
    """The convolve first approach also refuses to replace an existing mask"""
    existing = fits_dir / "image.scalemask.fits"
    fits.writeto(existing, np.full(SHAPE, 7.0))

    with pytest.raises(FileExistsError, match="already exists"):
        create_mask_from_fits(
            fits_image_path=fits_dir / "image.fits",
            masking_options=MaskingOptions(convolve_first=True),
        )

    assert np.all(fits.getdata(existing) == 7.0)


def test_fits_masking_convolve_first_threads(fits_dir: Path) -> None:
    """The mask does not depend on how many scales are processed at once"""
    rng = np.random.default_rng(seed=1)
    image = rng.normal(size=(200, 200)).astype(np.float32)
    image[80:120, 80:120] += 20.0
    image[30:33, 150:153] += 50.0
    header = fits.getheader(fits_dir / "image.fits")

    masks = []
    for threads in (1, 3):
        image_path = fits_dir / f"threads{threads}.fits"
        fits.writeto(image_path, data=image, header=header)
        names = create_mask_from_fits(
            fits_image_path=image_path,
            masking_options=MaskingOptions(
                convolve_first=True,
                seed_clip=1.2,
                flood_clip=0.5,
                mac_box_size=50,
                multiscale_scales=(0, 4, 8, 16),
                convolve_first_threads=threads,
            ),
        )
        assert names.scale_mask_fits is not None
        masks.append(fits.getdata(names.scale_mask_fits))

    assert np.any(masks[0] > 0), "Nothing was masked, so nothing is compared"
    assert len(np.unique(masks[0])) > 2, "Only one scale was masked"
    np.testing.assert_array_equal(masks[0], masks[1])


def test_fits_masking_convolve_first_threads_must_be_positive(fits_dir: Path) -> None:
    """Without a thread no scale would be processed"""
    with pytest.raises(AssertionError, match="threads"):
        create_mask_from_fits(
            fits_image_path=fits_dir / "image.fits",
            masking_options=MaskingOptions(
                convolve_first=True, convolve_first_threads=0
            ),
        )


def test_fits_masking_convolve_first_threads_keep_context(
    fits_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each scale sees the context of the caller, which is how the log messages
    of a scale reach a workflow manager"""
    caller: ContextVar[str] = ContextVar("caller", default="unset")
    caller.set("the caller")
    seen: dict[int, str] = {}

    def record_context(base_image, _fits_header, scale, **_):
        seen[scale] = caller.get()
        return np.zeros(base_image.shape, dtype=bool)

    monkeypatch.setattr(convolve, "_create_convolved_scale_mask", record_context)

    create_mask_from_fits(
        fits_image_path=fits_dir / "image.fits",
        masking_options=MaskingOptions(
            convolve_first=True,
            multiscale_scales=(0, 4, 8),
            convolve_first_threads=3,
        ),
    )

    assert seen == {0: "the caller", 4: "the caller", 8: "the caller"}
