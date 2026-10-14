from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from astropy.io import fits

from eye_patch.masking import _need_to_make_signal, create_mask_from_fits
from eye_patch.naming import FITSMaskNames
from eye_patch.options import MaskingOptions

if TYPE_CHECKING:
    from pathlib import Path

SHAPE = (100, 100)


def test_need_to_make_signal():
    """Ensure the conditions around creating a signal image make sense"""
    masking_options = MaskingOptions()
    assert _need_to_make_signal(masking_options=masking_options)

    masking_options = MaskingOptions(flood_fill=True, flood_fill_use_mac=True)
    assert not _need_to_make_signal(masking_options=masking_options)

    # Without the flood fill the mac is unused, and the base_snr_clip
    # needs a signal image rather than the raw image
    masking_options = MaskingOptions(flood_fill=False, flood_fill_use_mac=True)
    assert _need_to_make_signal(masking_options=masking_options)


def test_make_masking_options():
    """Just a dumb test to make sure the options structure is ok"""

    masking_options = MaskingOptions()

    assert masking_options.base_snr_clip != -1

    masking_options = masking_options.with_options(base_snr_clip=-1)  # type: ignore[arg-type]
    assert masking_options.base_snr_clip == -1


def test_fits_masking(fits_dir):
    masking_options = MaskingOptions(flood_fill=False)
    names = create_mask_from_fits(
        fits_image_path=fits_dir / "image.fits",
        fits_rms_path=fits_dir / "rms.fits",
        fits_bkg_path=fits_dir / "bkg.fits",
        masking_options=masking_options,
    )

    assert isinstance(names, FITSMaskNames)
    assert names.mask_fits is not None
    assert names.mask_fits.exists()
    assert names.signal_fits is None
    assert names.scale_mask_fits is None

    mask_data = fits.getdata(names.mask_fits)
    assert mask_data.dtype.kind == "f", "A bool mask cannot be written to FITS"
    valid = np.sum(mask_data)
    assert valid == np.prod(SHAPE)


@pytest.fixture
def beam_fits_dir(tmp_path: Path, beam_fits_header: fits.Header) -> Path:
    """As ``fits_dir``, but the image carries the beam keywords that
    ``beam_shape_erode`` needs to build its kernel."""
    beam_fits_dir = tmp_path / "beam_fits"
    beam_fits_dir.mkdir()

    fits.writeto(beam_fits_dir / "image.fits", np.ones(SHAPE), header=beam_fits_header)
    fits.writeto(beam_fits_dir / "rms.fits", np.ones(SHAPE) * 0.1)
    fits.writeto(beam_fits_dir / "bkg.fits", np.ones(SHAPE) * 0.5)

    return beam_fits_dir


def test_fits_masking_flood_fill_with_beam_erode(beam_fits_dir: Path) -> None:
    """The flood filled mask reaches ``beam_shape_erode``, which types its input
    as floats. This is the path the bool mask was travelling down."""
    masking_options = MaskingOptions(flood_fill=True, beam_shape_erode=True)
    names = create_mask_from_fits(
        fits_image_path=beam_fits_dir / "image.fits",
        fits_rms_path=beam_fits_dir / "rms.fits",
        fits_bkg_path=beam_fits_dir / "bkg.fits",
        masking_options=masking_options,
    )

    assert isinstance(names, FITSMaskNames)
    assert names.mask_fits is not None
    assert names.mask_fits.exists()

    mask_data = fits.getdata(names.mask_fits)
    assert mask_data.dtype.kind == "f", "A bool mask cannot be written to FITS"


def test_fits_masking_beam_erode_multi_scale_writes_scalemask(
    beam_fits_dir: Path,
) -> None:
    """Multiple erosion scales are bit-packed per pixel, and should be written
    to the scale mask rather than the plain mask. A compact source is used so
    that the larger scales erode away more of it, and each scale sets its own bit."""
    image_path = beam_fits_dir / "image.fits"
    header = fits.getheader(image_path)
    y, x = np.mgrid[: SHAPE[0], : SHAPE[1]]
    source = ((x - SHAPE[1] // 2) ** 2 + (y - SHAPE[0] // 2) ** 2) <= 10**2
    # Signal is 10 on the source and 0 elsewhere given the rms and bkg images
    fits.writeto(image_path, 0.5 + source * 1.0, header=header, overwrite=True)

    # Scales are wsclean scales, whose Gaussians have a FWHM of about 0.44 of the
    # scale, so they need to be well separated to erode this source differently
    masking_options = MaskingOptions(
        flood_fill=False, beam_shape_erode=True, beam_shape_erode_scales=(0, 8, 32)
    )
    names = create_mask_from_fits(
        fits_image_path=image_path,
        fits_rms_path=beam_fits_dir / "rms.fits",
        fits_bkg_path=beam_fits_dir / "bkg.fits",
        masking_options=masking_options,
    )

    assert names.mask_fits is None
    assert names.signal_fits is None
    assert names.scale_mask_fits is not None
    assert names.scale_mask_fits.exists()
    assert not (beam_fits_dir / "image.mask.fits").exists()

    mask_data = fits.getdata(names.scale_mask_fits)
    assert mask_data.dtype.kind == "f", "A bool mask cannot be written to FITS"
    # Larger scales are nested within the smaller ones: 0b001, 0b011, 0b111
    assert set(np.unique(mask_data)) == {0.0, 1.0, 3.0, 7.0}


def test_fits_masking_beam_erode_multi_scale_without_beam(fits_dir: Path) -> None:
    """Without beam parameters no erosion is performed, so the plain mask should
    be written rather than a misleading scale mask."""
    masking_options = MaskingOptions(
        flood_fill=False, beam_shape_erode=True, beam_shape_erode_scales=(0, 4, 8)
    )
    names = create_mask_from_fits(
        fits_image_path=fits_dir / "image.fits",
        fits_rms_path=fits_dir / "rms.fits",
        fits_bkg_path=fits_dir / "bkg.fits",
        masking_options=masking_options,
    )

    assert names.scale_mask_fits is None
    assert names.mask_fits is not None
    assert names.mask_fits.exists()
    assert not (fits_dir / "image.scalemask.fits").exists()

    mask_data = fits.getdata(names.mask_fits)
    assert set(np.unique(mask_data)) <= {0.0, 1.0}


def test_fits_masking_signal_not_needed_is_not_returned(fits_dir: Path) -> None:
    """The minimum absolute clip does not need a signal image, so even when one
    is requested it is not written and should not be returned."""
    masking_options = MaskingOptions(
        flood_fill=True, flood_fill_use_mac=True, flood_fill_positive_seed_clip=0.5
    )
    names = create_mask_from_fits(
        fits_image_path=fits_dir / "image.fits",
        fits_rms_path=None,
        fits_bkg_path=None,
        masking_options=masking_options,
        create_signal_fits=True,
    )

    assert names.signal_fits is None
    assert not (fits_dir / "image.signal.fits").exists()
    assert names.mask_fits is not None
    assert names.mask_fits.exists()


def test_fits_masking_flood_fill_writes_float_mask(fits_dir: Path) -> None:
    """The flood fill path builds its mask as a bool array, which ``fits.writeto``
    rejects. Reported by Beth via #14, and not caught earlier because every other
    mask test runs with ``flood_fill=False``."""
    masking_options = MaskingOptions(flood_fill=True)
    names = create_mask_from_fits(
        fits_image_path=fits_dir / "image.fits",
        fits_rms_path=fits_dir / "rms.fits",
        fits_bkg_path=fits_dir / "bkg.fits",
        masking_options=masking_options,
    )

    assert isinstance(names, FITSMaskNames)
    assert names.mask_fits is not None
    assert names.mask_fits.exists()

    mask_data = fits.getdata(names.mask_fits)
    assert mask_data.dtype.kind == "f", "A bool mask cannot be written to FITS"
    assert np.sum(mask_data) == np.prod(SHAPE)


def test_fits_masking_with_signal(fits_dir):
    masking_options = MaskingOptions(flood_fill=False)
    names = create_mask_from_fits(
        fits_image_path=fits_dir / "image.fits",
        fits_rms_path=fits_dir / "rms.fits",
        fits_bkg_path=fits_dir / "bkg.fits",
        masking_options=masking_options,
        create_signal_fits=True,
    )

    assert isinstance(names, FITSMaskNames)
    assert names.mask_fits is not None
    assert names.mask_fits.exists()
    assert names.signal_fits is not None
    assert names.signal_fits.exists()

    mask_data = fits.getdata(names.mask_fits)
    valid = np.sum(mask_data)
    assert valid == np.prod(SHAPE)

    signal_data = fits.getdata(names.signal_fits)
    assert np.allclose(signal_data, 5)


@pytest.mark.parametrize("convolve_first", [False, True])
def test_fits_masking_scales_as_fwhm(
    tmp_path: Path, beam_fits_header: fits.Header, convolve_first: bool
) -> None:
    """The comparison option should reach the erosion of both the beam erosion and
    the convolve first approaches, where it erodes the larger scale more"""
    yy, xx = np.mgrid[:200, :200]
    source = ((xx - 100) ** 2 + (yy - 100) ** 2) <= 40**2
    image = np.random.default_rng(1).normal(scale=0.01, size=(200, 200)) + source
    fits.writeto(tmp_path / "image.fits", image, header=beam_fits_header)
    fits.writeto(tmp_path / "rms.fits", np.full((200, 200), 0.01))
    fits.writeto(tmp_path / "bkg.fits", np.zeros((200, 200)))

    largest_scale_pixels = []
    for scales_as_fwhm in (False, True):
        names = create_mask_from_fits(
            fits_image_path=tmp_path / "image.fits",
            fits_rms_path=tmp_path / "rms.fits",
            fits_bkg_path=tmp_path / "bkg.fits",
            masking_options=MaskingOptions(
                convolve_first=convolve_first,
                beam_shape_erode=True,
                beam_shape_erode_scales=(0, 32),
                beam_shape_erode_scales_as_fwhm=scales_as_fwhm,
            ),
        )
        assert names.scale_mask_fits is not None
        scale_mask = fits.getdata(names.scale_mask_fits).astype(int)
        largest_scale_pixels.append(np.sum((scale_mask >> 1) & 1))

    assert 0 < largest_scale_pixels[1] < largest_scale_pixels[0]


def test_fits_masking_mac_without_flood_fill_uses_signal(fits_dir: Path) -> None:
    """The mac is only used with the flood fill, so the base_snr_clip should be
    applied to the signal image (5 everywhere) rather than the raw image (1)"""
    masking_options = MaskingOptions(flood_fill=False, flood_fill_use_mac=True)
    names = create_mask_from_fits(
        fits_image_path=fits_dir / "image.fits",
        fits_rms_path=fits_dir / "rms.fits",
        fits_bkg_path=fits_dir / "bkg.fits",
        masking_options=masking_options,
    )

    assert names.mask_fits is not None
    assert np.sum(fits.getdata(names.mask_fits)) == np.prod(SHAPE)


def test_fits_masking_mac_without_flood_fill_needs_rms_bkg(fits_dir: Path) -> None:
    """Without the flood fill a signal image is needed. A missing RMS and
    background should raise an error that points to the flood fill setting."""
    masking_options = MaskingOptions(flood_fill=False, flood_fill_use_mac=True)

    with pytest.raises(ValueError, match="flood_fill=True"):
        create_mask_from_fits(
            fits_image_path=fits_dir / "image.fits",
            fits_rms_path=None,
            fits_bkg_path=None,
            masking_options=masking_options,
        )
