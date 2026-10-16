from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from astropy.io import fits

from eye_patch.mac import minimum_absolute_clip
from eye_patch.masking import (
    _create_island_mask,
    _need_to_make_signal,
    create_mask_from_fits,
)
from eye_patch.naming import FITSMaskNames
from eye_patch.options import MaskingOptions

if TYPE_CHECKING:
    from pathlib import Path

SHAPE = (100, 100)


def test_need_to_make_signal():
    """Ensure the conditions around creating a signal image make sense"""
    masking_options = MaskingOptions()
    assert _need_to_make_signal(masking_options=masking_options)

    # The mac operates on the image itself, with or without the flood fill
    for flood_fill in (True, False):
        masking_options = MaskingOptions(flood_fill=flood_fill, use_mac=True)
        assert not _need_to_make_signal(masking_options=masking_options)


def test_make_masking_options():
    """Just a dumb test to make sure the options structure is ok"""

    masking_options = MaskingOptions()

    assert masking_options.base_clip != -1

    masking_options = masking_options.with_options(base_clip=-1)  # type: ignore[arg-type]
    assert masking_options.base_clip == -1


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
        flood_fill=False, beam_shape_erode=True, multiscale_scales=(0, 8, 32)
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
        flood_fill=False, beam_shape_erode=True, multiscale_scales=(0, 4, 8)
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
    masking_options = MaskingOptions(flood_fill=True, use_mac=True, seed_clip=0.5)
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
                multiscale_scales=(0, 32),
                beam_shape_erode_scales_as_fwhm=scales_as_fwhm,
            ),
        )
        assert names.scale_mask_fits is not None
        scale_mask = fits.getdata(names.scale_mask_fits).astype(int)
        largest_scale_pixels.append(np.sum((scale_mask >> 1) & 1))

    assert 0 < largest_scale_pixels[1] < largest_scale_pixels[0]


def test_fits_masking_mac_without_flood_fill(tmp_path: Path) -> None:
    """The mac can be used without the flood fill, where each pixel is clipped at
    base_clip times its clipping level. As the image itself is clipped, no RMS or
    background images are needed."""
    image = np.random.default_rng(1).normal(size=SHAPE)
    image[40:60, 40:60] += 20.0
    image_path = tmp_path / "image.fits"
    fits.writeto(image_path, image)

    names = create_mask_from_fits(
        fits_image_path=image_path,
        fits_rms_path=None,
        fits_bkg_path=None,
        masking_options=MaskingOptions(use_mac=True, base_clip=2.0, mac_box_size=30),
    )

    assert names.mask_fits is not None
    mask_data = fits.getdata(names.mask_fits)
    assert mask_data.dtype.itemsize == 4, "Masks are written as 32-bit floats"
    expected = minimum_absolute_clip(image=image, increase_factor=2.0, box_size=30)
    assert np.array_equal(mask_data == 1, expected)
    assert np.all(expected[40:60, 40:60])


def test_fits_masking_grows_low_snr_islands_without_flood_fill(tmp_path: Path) -> None:
    """Islands of faint and diffuse emission are added to the mask whether or not
    the flood fill is used"""
    yy, xx = np.mgrid[:200, :200]
    diffuse = ((xx - 100) ** 2 + (yy - 100) ** 2) <= 40**2
    image = np.random.default_rng(2).normal(size=(200, 200)) + diffuse * 2.5
    fits.writeto(tmp_path / "image.fits", image)
    fits.writeto(tmp_path / "rms.fits", np.ones((200, 200)))
    fits.writeto(tmp_path / "bkg.fits", np.zeros((200, 200)))

    masks = []
    for grow_low_snr_island in (False, True):
        names = create_mask_from_fits(
            fits_image_path=tmp_path / "image.fits",
            fits_rms_path=tmp_path / "rms.fits",
            fits_bkg_path=tmp_path / "bkg.fits",
            masking_options=MaskingOptions(
                base_clip=5.0, grow_low_snr_island=grow_low_snr_island
            ),
        )
        assert names.mask_fits is not None
        masks.append(fits.getdata(names.mask_fits) > 0)
    without_growth, with_growth = masks

    assert np.mean(without_growth[diffuse]) < 0.1
    assert np.mean(with_growth[diffuse]) > 0.95
    assert np.mean(with_growth[~diffuse]) < 0.02


@pytest.mark.parametrize("flood_fill", [True, False])
def test_create_island_mask_mac_grows_low_snr_islands(flood_fill: bool) -> None:
    """With the minimum absolute clip the image is not a signal map. Low SNR islands
    should be found using the clip as an increase factor of the same clipping level,
    rather than comparing a significance to the raw pixel values, whether or not the
    flood fill is used."""
    noise = 1e-4  # An image in Jy/beam rather than a signal map
    yy, xx = np.mgrid[:400, :400]
    diffuse = ((xx - 200) ** 2 + (yy - 200) ** 2) <= 60**2
    image = (
        np.random.default_rng(3).normal(scale=noise, size=(400, 400))
        + diffuse * 2.5 * noise
    )

    without_growth, with_growth = (
        _create_island_mask(
            base_image=image,
            masking_options=MaskingOptions(
                flood_fill=flood_fill,
                use_mac=True,
                grow_low_snr_island=grow_low_snr_island,
                grow_low_snr_island_clip=0.5,
            ),
        )
        for grow_low_snr_island in (False, True)
    )

    assert np.mean(without_growth[diffuse]) < 0.1
    assert np.mean(with_growth[diffuse]) > 0.95
    assert np.mean(with_growth[~diffuse]) < 0.02


# Each approach to masking, which should all keep degenerate axes of the image
DEGENERATE_AXES_OPTIONS: dict[str, dict[str, Any]] = {
    "snr_clip": {},
    "mac_clip": {"use_mac": True},
    "mac_clip_adaptive": {"use_mac": True, "mac_adaptive_max_depth": 2},
    "flood_fill_snr": {"flood_fill": True},
    "flood_fill_mac": {"flood_fill": True, "use_mac": True},
    "low_snr_islands": {"grow_low_snr_island": True},
    "low_snr_islands_mac": {
        "use_mac": True,
        "grow_low_snr_island": True,
        "grow_low_snr_island_clip": 0.5,
    },
    "flood_fill_low_snr_islands": {"flood_fill": True, "grow_low_snr_island": True},
    "beam_erode": {"flood_fill": True, "beam_shape_erode": True},
    "beam_erode_multi_scale": {
        "flood_fill": True,
        "beam_shape_erode": True,
        "multiscale_scales": (0, 8),
    },
    "convolve_first": {"convolve_first": True},
    "convolve_first_multi_scale": {"convolve_first": True, "multiscale_scales": (0, 8)},
}


@pytest.mark.parametrize(
    "options",
    list(DEGENERATE_AXES_OPTIONS.values()),
    ids=list(DEGENERATE_AXES_OPTIONS),
)
def test_fits_masking_with_degenerate_axes(
    tmp_path: Path, beam_fits_header: fits.Header, options: dict[str, Any]
) -> None:
    """An image with degenerate frequency and Stokes axes, as written by wsclean,
    should be masked as its two-dimensional equivalent, with every product taking
    the shape of the image"""
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    image = (
        np.random.default_rng(4).normal(size=SHAPE)
        + (((xx - 30) ** 2 + (yy - 30) ** 2) <= 8**2) * 50.0
        + (((xx - 65) ** 2 + (yy - 65) ** 2) <= 25**2) * 2.5
    )

    products = {}
    for shape in (SHAPE, (1, 1, *SHAPE)):
        directory = tmp_path / f"{len(shape)}d"
        directory.mkdir()
        fits.writeto(
            directory / "image.fits", image.reshape(shape), header=beam_fits_header
        )
        fits.writeto(directory / "rms.fits", np.ones(shape))
        fits.writeto(directory / "bkg.fits", np.zeros(shape))
        names = create_mask_from_fits(
            fits_image_path=directory / "image.fits",
            fits_rms_path=directory / "rms.fits",
            fits_bkg_path=directory / "bkg.fits",
            masking_options=MaskingOptions(**options),
            create_signal_fits=True,
        )
        products[len(shape)] = {
            field: fits.getdata(path)
            for field, path in names._asdict().items()
            if path is not None
        }
        assert all(data.shape == shape for data in products[len(shape)].values())

    assert products[2].keys() == products[4].keys()
    for field, data in products[2].items():
        assert np.array_equal(data, np.squeeze(products[4][field]))
    mask_field = "scale_mask_fits" if "scale_mask_fits" in products[2] else "mask_fits"
    assert np.any(products[2][mask_field])
