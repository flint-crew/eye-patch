from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from eye_patch.cli import cli
from eye_patch.extract import extract_beam_mask_from_mosaic
from eye_patch.naming import FITSMaskNames

if TYPE_CHECKING:
    from pathlib import Path

SHAPE = (100, 100)


def _celestial_header(crpix: tuple[float, float], naxis: int = 2) -> fits.Header:
    """A simple celestial header that only differs in its reference pixel. With a
    ``naxis`` of 4 frequency and Stokes axes are added, as written by wsclean."""
    wcs = WCS(naxis=naxis)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"][:naxis]
    wcs.wcs.crval = [150.0, -30.0, 1.4e9, 1.0][:naxis]
    wcs.wcs.cdelt = [-0.001, 0.001, 1e6, 1.0][:naxis]
    wcs.wcs.crpix = [*crpix, 1.0, 1.0][:naxis]
    return wcs.to_header()


def test_extract_beam_mask_from_mosaic_outside_footprint(tmp_path: Path) -> None:
    """Pixels of the template image outside of the mosaic footprint are NaN after
    reprojection, which used to be cast to a large non-zero integer"""
    mosaic_path = tmp_path / "mosaic.mask.fits"
    fits.writeto(
        mosaic_path,
        np.ones(SHAPE, dtype=np.int32),
        header=_celestial_header(crpix=(50, 50)),
    )

    # Shifted so that half of the template falls outside of the mosaic
    template_path = tmp_path / "beam.fits"
    fits.writeto(
        template_path, np.zeros(SHAPE), header=_celestial_header(crpix=(100, 50))
    )

    for _ in range(2):  # Running again should overwrite rather than raise
        names = extract_beam_mask_from_mosaic(
            fits_beam_image_path=template_path,
            fits_mosaic_mask_names=FITSMaskNames(mask_fits=mosaic_path),
        )

    assert names.mask_fits is not None
    mask_data = fits.getdata(names.mask_fits)
    assert set(np.unique(mask_data)) == {0, 1}
    assert 0.3 < np.mean(mask_data) < 0.7


@pytest.fixture
def mosaic_scale_mask_dir(tmp_path: Path) -> Path:
    """A per-scale mosaic mask, and a template image with half of its pixels
    outside of the mosaic. The template's right half covers the mosaic's left half."""
    mosaic_scale_mask = np.zeros(SHAPE, dtype=np.float32)
    mosaic_scale_mask[:, :50] = 1
    mosaic_scale_mask[20:80, 10:40] = 3
    mosaic_scale_mask[40:60, 20:30] = 7
    fits.writeto(
        tmp_path / "mosaic.scalemask.fits",
        mosaic_scale_mask,
        header=_celestial_header(crpix=(50, 50)),
    )
    fits.writeto(
        tmp_path / "beam.fits",
        np.zeros(SHAPE),
        header=_celestial_header(crpix=(100, 50)),
    )
    return tmp_path


def test_extract_beam_scale_mask_from_mosaic(mosaic_scale_mask_dir: Path) -> None:
    """A per-scale mask should be extracted without clipping its bitmapped values"""
    names = extract_beam_mask_from_mosaic(
        fits_beam_image_path=mosaic_scale_mask_dir / "beam.fits",
        fits_mosaic_mask_names=FITSMaskNames(
            scale_mask_fits=mosaic_scale_mask_dir / "mosaic.scalemask.fits"
        ),
    )

    assert names.mask_fits is None
    assert names.scale_mask_fits is not None
    assert names.scale_mask_fits == mosaic_scale_mask_dir / "beam.scalemask.fits"

    scale_mask_data = fits.getdata(names.scale_mask_fits)
    assert set(np.unique(scale_mask_data)) == {0.0, 1.0, 3.0, 7.0}
    # Pixels outside of the mosaic footprint
    assert np.all(scale_mask_data[:, :45] == 0)


def test_extract_beam_mask_from_mosaic_without_masks(
    mosaic_scale_mask_dir: Path,
) -> None:
    """There needs to be at least one mask to extract from"""
    with pytest.raises(ValueError, match="mask_fits or scale_mask_fits"):
        extract_beam_mask_from_mosaic(
            fits_beam_image_path=mosaic_scale_mask_dir / "beam.fits",
            fits_mosaic_mask_names=FITSMaskNames(),
        )


def test_cli_extract_beam_scale_mask(
    mosaic_scale_mask_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI treats a mosaic ending in .scalemask.fits as a per-scale mask"""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eye-patch",
            "extractmask",
            str(mosaic_scale_mask_dir / "beam.fits"),
            str(mosaic_scale_mask_dir / "mosaic.scalemask.fits"),
        ],
    )
    cli()

    assert not (mosaic_scale_mask_dir / "beam.mask.fits").exists()
    scale_mask_data = fits.getdata(mosaic_scale_mask_dir / "beam.scalemask.fits")
    assert np.max(scale_mask_data) == 7.0


@pytest.mark.parametrize("mosaic_naxis", [2, 4])
def test_extract_beam_masks_have_the_template_shape(
    tmp_path: Path, mosaic_naxis: int
) -> None:
    """Extracted masks take the shape of the template, which may carry degenerate
    frequency and Stokes axes, with the same values as for a two-dimensional
    template. The template's right half covers the mosaic's left half."""
    mosaic_scale_mask = np.zeros(SHAPE, dtype=np.float32)
    mosaic_scale_mask[:, :50] = 1
    mosaic_scale_mask[20:80, 10:40] = 3
    mosaic_shape = SHAPE if mosaic_naxis == 2 else (1, 1, *SHAPE)
    mosaic_header = _celestial_header(crpix=(50, 50), naxis=mosaic_naxis)
    mosaic_names = FITSMaskNames(
        mask_fits=tmp_path / "mosaic.mask.fits",
        scale_mask_fits=tmp_path / "mosaic.scalemask.fits",
    )
    fits.writeto(
        mosaic_names.mask_fits,
        np.clip(mosaic_scale_mask, 0, 1).reshape(mosaic_shape),
        header=mosaic_header,
    )
    fits.writeto(
        mosaic_names.scale_mask_fits,
        mosaic_scale_mask.reshape(mosaic_shape),
        header=mosaic_header,
    )

    extracted = {}
    for template_shape in (SHAPE, (1, 1, *SHAPE)):
        template_dir = tmp_path / f"template_{len(template_shape)}d"
        template_dir.mkdir()
        fits.writeto(
            template_dir / "beam.fits",
            np.zeros(template_shape),
            header=_celestial_header(crpix=(100, 50), naxis=len(template_shape)),
        )
        names = extract_beam_mask_from_mosaic(
            fits_beam_image_path=template_dir / "beam.fits",
            fits_mosaic_mask_names=mosaic_names,
        )
        assert names.mask_fits is not None
        assert names.scale_mask_fits is not None
        masks = (fits.getdata(names.mask_fits), fits.getdata(names.scale_mask_fits))
        assert all(mask.shape == template_shape for mask in masks)
        extracted[len(template_shape)] = masks

    for mask_2d, mask_4d in zip(extracted[2], extracted[4], strict=True):
        assert np.array_equal(mask_2d, np.squeeze(mask_4d))
    assert np.max(extracted[2][1]) == 3.0
