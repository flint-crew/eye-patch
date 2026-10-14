from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from capn_crunch import create_options_from_parser
from radio_beam import Beam
from scipy.ndimage import binary_dilation, binary_erosion, label, minimum_filter
from scipy.signal import fftconvolve

from eye_patch.cli import cli, get_parser
from eye_patch.convolve import convolve_image_by_scale
from eye_patch.erosion import (
    beam_shape_erode,
    create_beam_mask_kernel,
    create_multi_scale_erosion,
    fft_binary_erosion,
)
from eye_patch.extract import extract_beam_mask_from_mosaic
from eye_patch.flood_fill import (
    _flood_fill,
    _verify_set_positive_seed_clip,
    create_flood_fill_mask,
    grow_low_snr_mask,
    reverse_negative_flood_fill,
)
from eye_patch.logging import CustomFormatter, logger
from eye_patch.mac import (
    create_boxcar_skew_mask,
    minimum_absolute_clip,
    minimum_absolute_clip_threshold,
)
from eye_patch.masking import _need_to_make_signal, create_snr_mask_from_fits
from eye_patch.naming import FITSMaskNames
from eye_patch.options import MaskingOptions
from eye_patch.snr import _create_signal_from_rmsbkg

SHAPE = (100, 100)


def test_adaptive_minimum_absolute_clip():
    """Activate the adaptive mode of the mac"""

    image = np.ones((2000, 2000)) * -1.2
    image[100:200, 100:200] = 1.0
    mask = minimum_absolute_clip(
        image=image,
        box_size=50,
        increase_factor=1.1,
        adaptive_max_depth=1,
        adaptive_box_step=4.0,
        adaptive_skew_delta=0.2,
    )
    assert np.all(~mask)

    image = np.ones((2000, 2000)) * -1.2
    image[100:200, 100:200] = 1.0
    image[110, 130] = 2000
    mask = minimum_absolute_clip(
        image=image,
        box_size=50,
        increase_factor=1.1,
        adaptive_max_depth=1,
        adaptive_box_step=4.0,
        adaptive_skew_delta=0.2,
    )
    assert np.sum(mask) == 1


def test_minimum_absolute_clip_simple():
    """The minimum absolute clipping process accepts an image
    and returns a mask.  At its heart it applies a ``minimum_filter``
    from scipy."""

    image = np.ones(SHAPE) * -1.0
    image[10, 10] = -2
    mac_mask = minimum_absolute_clip(image=image, box_size=10)
    assert np.all(~mac_mask)

    image[12, 12] = 5
    mac_mask = minimum_absolute_clip(image=image)
    assert np.sum(mac_mask) == 1

    image[12, 12] = 0
    mac_mask = minimum_absolute_clip(image=image)
    assert np.all(~mac_mask)

    image[12, 12] = 5
    mac_mask = minimum_absolute_clip(image=image, increase_factor=3)
    assert np.all(~mac_mask)


def test_create_signal_from_rmsbkg():
    """Make sure the operations around the creation of a signal image from
    rms / bkg images follows and handles paths vs numpy arrays"""
    shape = (100, 100)
    image = np.zeros(shape) + 10
    bkg = np.ones(shape)
    rms = np.zeros(shape) + 3

    signal = _create_signal_from_rmsbkg(image=image, rms=rms, bkg=bkg)
    assert np.all(signal == 3.0)


def test_create_signal_from_rmsbkg_with_fits(tmpdir):
    """Make sure the operations around the creation of a signal image from
    rms / bkg images follows and handles paths vs numpy arrays. Unlike the
    above this uses fits for some inputs"""
    out_path = Path(tmpdir) / "fits"
    out_path.mkdir(parents=True, exist_ok=True)

    shape = (100, 100)
    image = np.zeros(shape) + 10
    image_fits = Path(out_path) / "image.fits"
    fits.writeto(image_fits, data=image)

    bkg = np.ones(shape)
    rms = np.zeros(shape) + 3
    rms_fits = Path(out_path) / "rms.fits"
    fits.writeto(rms_fits, data=rms)

    signal = _create_signal_from_rmsbkg(image=image_fits, rms=rms_fits, bkg=bkg)
    assert np.all(signal == 3.0)


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


def test_arg_parser_cli_and_masking_options():
    """See if the CLI parser is constructed with correct set of
    options which are properly converted to a MaskingOptions object"""
    parser = get_parser()
    args = parser.parse_args(
        args=[
            "mask",
            "img",
            "--rms-fits",
            "rms",
            "--bkg-fits",
            "bkg",
            "--flood-fill",
            "--flood-fill-positive-seed-clip",
            "10",
            "--flood-fill-positive-flood-clip",
            "1.",
            "--flood-fill-use-mac",
            "--flood-fill-use-mac-box-size",
            "100",
        ]
    )
    masking_options = create_options_from_parser(
        parser_namespace=args, options_class=MaskingOptions
    )
    assert isinstance(masking_options, MaskingOptions)
    assert masking_options.flood_fill
    assert masking_options.flood_fill_use_mac
    assert masking_options.flood_fill_positive_seed_clip == 10.0
    assert masking_options.flood_fill_positive_flood_clip == 1.0


def test_arg_parser_cli_and_masking_options_for_multiscale():
    """See if the CLI parser is constructed with correct set of
    options which are properly converted to a MaskingOptions object. Here
    want to make sure that the multiscale erosion are picked up"""
    parser = get_parser()
    args = parser.parse_args(
        args=[
            "mask",
            "img",
            "--beam-shape-erode-scales",
            "0",
            "1",
            "2",
            "4",
            "--rms-fits",
            "rms",
            "--bkg-fits",
            "bkg",
            "--flood-fill",
            "--flood-fill-positive-seed-clip",
            "10",
            "--flood-fill-positive-flood-clip",
            "1.",
            "--flood-fill-use-mac",
            "--flood-fill-use-mac-box-size",
            "100",
        ]
    )
    masking_options = create_options_from_parser(
        parser_namespace=args, options_class=MaskingOptions
    )
    assert isinstance(masking_options, MaskingOptions)
    assert isinstance(masking_options.beam_shape_erode_scales, tuple)
    assert masking_options.beam_shape_erode_scales == (0, 1, 2, 4)
    assert not masking_options.beam_shape_erode_scales_as_fwhm


def test_arg_parser_scales_as_fwhm() -> None:
    """The option to use scales as a FWHM, as in earlier versions, is available
    from the CLI for comparisons"""
    args = get_parser().parse_args(
        args=["mask", "img", "--beam-shape-erode-scales-as-fwhm"]
    )
    masking_options = create_options_from_parser(
        parser_namespace=args, options_class=MaskingOptions
    )
    assert masking_options.beam_shape_erode_scales_as_fwhm


@pytest.fixture
def beam_fits_header() -> fits.Header:
    return fits.Header(
        {
            "CDELT1": -0.000694444444444444,
            "CDELT2": 0.000694444444444444,
            "BMAJ": 0.00340540107886635,
            "BMIN": 0.00283268735470751,
            "BPA": 74.6618858613889,
        }
    )


def test_create_beam_mask_kernel(beam_fits_header):
    """See whether the beam kernel creation mask can return a correct mask"""
    fits_header = beam_fits_header
    mask_1 = create_beam_mask_kernel(fits_header=fits_header)
    assert mask_1.shape == (100, 100)
    assert np.sum(mask_1) == 12

    mask_2 = create_beam_mask_kernel(fits_header=fits_header, minimum_response=0.1)
    assert mask_2.shape == (100, 100)
    assert np.sum(mask_2) == 52

    fits_header = fits.Header(
        {
            "CDELT1": -0.000694444444444444,
            "CDELT2": 0.2000694444444444444,
            "BMAJ": 0.00340540107886635,
            "BMIN": 0.00283268735470751,
            "BPA": 74.6618858613889,
        }
    )
    with pytest.raises(AssertionError):
        create_beam_mask_kernel(fits_header=fits_header)

    fits_header = fits.Header(
        {
            "CDELT1": -0.000694444444444444,
            "BMAJ": 0.00340540107886635,
            "BMIN": 0.00283268735470751,
            "BPA": 74.6618858613889,
        }
    )
    with pytest.raises(KeyError):
        create_beam_mask_kernel(fits_header=fits_header)


def test_create_beam_mask_kernel_with_scales(beam_fits_header):
    """See whether the beam kernel creation mask can return a correct mask. Here
    we also test for different scales."""

    fits_header = beam_fits_header
    mask_1 = create_beam_mask_kernel(fits_header=fits_header)
    assert mask_1.shape == (100, 100)
    assert np.sum(mask_1) == 12

    mask_2 = create_beam_mask_kernel(fits_header=fits_header, pixel_scale=0)
    assert mask_2.shape == (100, 100)
    assert np.sum(mask_2) == 12

    assert np.all(mask_1 == mask_2)

    mask_3 = create_beam_mask_kernel(fits_header=fits_header, pixel_scale=4)
    assert mask_3.shape == (100, 100)
    assert np.sum(mask_3) == 22

    assert not np.all(mask_1 == mask_3)

    mask_4 = create_beam_mask_kernel(fits_header=fits_header, pixel_scale=32)
    assert mask_4.shape == (100, 100)
    assert np.sum(mask_4) == 608

    assert not np.all(mask_1 == mask_4)

    # With these options the output shape should be
    # the same as the input shape
    test_mask = np.zeros((200, 440))
    mask_5 = create_beam_mask_kernel(
        fits_header=fits_header,
        pixel_scale=32,
        auto_resize=False,
        kernel_size=test_mask.shape,
    )
    assert test_mask.shape == mask_5.shape


def test_create_beam_mask_kernel_with_scales_resize(beam_fits_header):
    """See whether the beam kernel creation mask can return a correct mask. Here
    we also test for different scales. This is test the resize component of the
    kernel evaluation"""

    fits_header = beam_fits_header
    mask_1 = create_beam_mask_kernel(
        fits_header=fits_header, auto_resize=False, pixel_scale=512
    )
    assert np.all(mask_1 == 1)

    fits_header = beam_fits_header
    mask_2 = create_beam_mask_kernel(
        fits_header=fits_header, auto_resize=True, pixel_scale=512
    )
    assert mask_1.shape != mask_2.shape
    assert 0 in mask_2
    # Resized kernels are odd so the beam stays centred on a pixel
    assert all(size % 2 == 1 for size in mask_2.shape)


def test_beam_shape_erode(beam_fits_header):
    """Ensure that the beam shape erosion approach works. This should drop out pixels
    should the beam shape structure connectivity be statisifed"""
    fits_header = beam_fits_header

    mask = np.zeros((500, 500)).astype(bool)

    mask[300, 300] = True
    assert np.sum(mask) == 1

    new_mask = beam_shape_erode(mask=mask, fits_header=fits_header)
    assert np.sum(new_mask) == 0

    mask = np.zeros((200, 200)).astype(bool)
    mask[100:130, 100:130] = True
    assert np.sum(mask) == 900
    erode_mask = beam_shape_erode(mask=mask, fits_header=fits_header)
    # The centred beam structure spans 3 rows and 5 columns, so the eroded
    # square is 28 x 26 pixels and remains centred on the original square
    assert np.sum(erode_mask) == 728
    rows, cols = np.nonzero(erode_mask)
    assert np.isclose(np.mean(rows), 114.5)
    assert np.isclose(np.mean(cols), 114.5)


def test_beam_shape_erode_scales(beam_fits_header):
    """Ensure that the beam shape erosion approach works. This should drop out pixels
    should the beam shape structure connectivity be statisifed"""
    fits_header = beam_fits_header

    # There should be nothing that passes this
    mask = np.zeros((500, 500)).astype(bool)
    mask[300, 300] = True
    assert np.sum(mask) == 1

    erode_mask = beam_shape_erode(
        mask=mask, fits_header=fits_header, scales=[0, 2, 4, 8]
    )
    assert erode_mask.dtype != mask.dtype
    assert np.max(erode_mask) == 0.0

    # All scales should pass
    mask = np.zeros((500, 500)).astype(bool)
    mask[300:330, 300:330] = True
    assert np.sum(mask) == 900

    erode_mask = beam_shape_erode(
        mask=mask, fits_header=fits_header, scales=[0, 2, 4, 8]
    )
    assert erode_mask.dtype != mask.dtype
    assert np.max(erode_mask) == 15

    # All scales should pass
    mask = np.zeros((500, 500)).astype(bool)
    mask[300:310, 300:310] = True
    assert np.sum(mask) == 100

    erode_mask = beam_shape_erode(
        mask=mask, fits_header=fits_header, scales=[0, 2, 32, 64, 128]
    )
    assert erode_mask.dtype != mask.dtype
    assert np.max(erode_mask) == 3


def test_fft_binary_erosion() -> None:
    """Basic test for binaruy erosion done using the FFT approach"""
    base_shape = (1000, 1000)
    mask = np.zeros(base_shape)
    mask[450:550, 450:550] = 1

    kernel = np.zeros(base_shape)
    kernel[490:510, 490:510] = 1
    eroded = fft_binary_erosion(mask=mask, kernel=kernel.astype(bool))
    assert np.sum(eroded) == 6561

    kernel = np.zeros(base_shape)
    kernel[450:550, 450:550] = 1
    eroded = fft_binary_erosion(mask=mask, kernel=kernel.astype(bool))
    assert np.sum(eroded) == 1
    assert np.unravel_index(np.argmax(eroded), base_shape) == (
        np.int64(500),
        np.int64(500),
    )

    kernel = np.zeros(base_shape)
    kernel[400:600, 400:600] = 1
    eroded = fft_binary_erosion(mask=mask, kernel=kernel.astype(bool))
    assert np.sum(eroded) == 0


def test_beam_shape_erode_nobeam():
    """Ensure that the beam shape erosion approach works. This should drop out pixels
    should the beam shape structure connectivity be statisifed. This should simply return
    the input array since there is no beam information"""
    fits_header = fits.Header(
        {
            "CDELT1": -0.000694444444444444,
            "CDELT2": 0.000694444444444444,
        }
    )

    mask = np.zeros((500, 500)).astype(bool)

    mask[300, 300] = True
    assert np.sum(mask) == 1
    new_mask = beam_shape_erode(mask=mask, fits_header=fits_header)
    assert new_mask is mask
    assert np.sum(new_mask) == 1


@pytest.fixture
def fits_dir(tmpdir):
    fits_dir = Path(tmpdir) / "fits"
    fits_dir.mkdir()

    shape = SHAPE

    img_data = np.ones(shape)
    bkg_data = np.ones(shape) * 0.5
    rms_data = np.ones(shape) * 0.1

    fits.writeto(fits_dir / "image.fits", img_data)
    fits.writeto(fits_dir / "rms.fits", rms_data)
    fits.writeto(fits_dir / "bkg.fits", bkg_data)

    return fits_dir


def test_make_masking_options():
    """Just a dumb test to make sure the options structure is ok"""

    masking_options = MaskingOptions()

    assert masking_options.base_snr_clip != -1

    masking_options = masking_options.with_options(base_snr_clip=-1)  # type: ignore[arg-type]
    assert masking_options.base_snr_clip == -1


def test_verify_set_seed_clip():
    """Make sure the flood seed clip handles items above all possible values"""
    signal = np.ones((100, 100)) * 10.0

    flood_clip = _verify_set_positive_seed_clip(signal=signal, positive_seed_clip=9.0)

    assert flood_clip == 9.0
    flood_clip = _verify_set_positive_seed_clip(
        signal=signal, positive_seed_clip=999999.0
    )
    assert flood_clip == 9.0


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


def test_fits_masking(fits_dir):
    masking_options = MaskingOptions(flood_fill=False)
    names = create_snr_mask_from_fits(
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
    names = create_snr_mask_from_fits(
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
    names = create_snr_mask_from_fits(
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
    names = create_snr_mask_from_fits(
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
    names = create_snr_mask_from_fits(
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
    names = create_snr_mask_from_fits(
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
    names = create_snr_mask_from_fits(
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


def test_beam_shape_erode_is_not_shifted(beam_fits_header: fits.Header) -> None:
    """Even sized beam kernels have no centre pixel, which shifted every eroded
    mask by half a pixel. Each scale should stay centred on the source."""
    scales = (0, 4, 8, 16)
    yy, xx = np.mgrid[:256, :256]
    mask = (((xx - 128) ** 2 + (yy - 128) ** 2) <= 40**2).astype(float)

    eroded_mask = beam_shape_erode(
        mask=mask, fits_header=beam_fits_header, scales=scales
    ).astype(int)

    for index in range(len(scales)):
        scale_mask = (eroded_mask >> index) & 1
        assert np.sum(scale_mask) > 0
        assert np.allclose(_disk_centroid(image=scale_mask), (128.0, 128.0))


def _wsclean_gaussian_kernel(scale: int) -> np.ndarray:
    """The Gaussian kernel that wsclean (via radler) forms for a multi-scale scale,
    with a sigma of 3/16 of the scale, an odd size and centred on a pixel"""
    sigma = scale * 3.0 / 16.0
    size = int(np.ceil(sigma * 12.0 / 2.0)) * 2 + 1
    profile = np.exp(-((np.arange(size) - size // 2) ** 2) / (2 * sigma**2))
    kernel = np.outer(profile, profile)
    normalised_kernel: np.ndarray = kernel / np.sum(kernel)
    return normalised_kernel


@pytest.mark.parametrize("scale", [8, 16, 32])
def test_multi_scale_erosion_matches_wsclean_gaussian(
    beam_fits_header: fits.Header, scale: int
) -> None:
    """The erosion at a scale should use the restoring beam convolved with the
    Gaussian that wsclean uses for that scale. Treating the scale as a FWHM made
    the erosion structure about 2.3 times too wide."""
    beam = Beam.from_fits_header(beam_fits_header)
    pixel_size = abs(beam_fits_header["CDELT1"]) * u.deg
    beam_kernel = beam.as_kernel(pixscale=pixel_size, x_size=101, y_size=101).array
    beam_scale_response = fftconvolve(
        beam_kernel, _wsclean_gaussian_kernel(scale=scale), mode="same"
    )
    structure = beam_scale_response > 0.6 * np.max(beam_scale_response)

    yy, xx = np.mgrid[:256, :256]
    mask = ((xx - 128) ** 2 + (yy - 128) ** 2) <= 40**2

    expected_mask = binary_erosion(input=mask, structure=structure)
    eroded_mask = create_multi_scale_erosion(
        mask=mask, fits_header=beam_fits_header, scale=scale
    )

    assert np.array_equal(eroded_mask, expected_mask)


@pytest.mark.parametrize("scale", [8, 16, 32])
def test_multi_scale_erosion_scale_is_fwhm(
    beam_fits_header: fits.Header, scale: int
) -> None:
    """The comparison option uses the scale as the FWHM of the gaussian, as in
    earlier versions, which erodes more than the wsclean gaussian of the scale"""
    yy, xx = np.mgrid[:256, :256]
    mask = ((xx - 128) ** 2 + (yy - 128) ** 2) <= 40**2

    structure = create_beam_mask_kernel(
        fits_header=beam_fits_header,
        kernel_size=101,
        pixel_scale=scale,
        auto_resize=False,
    )
    expected_mask = binary_erosion(input=mask, structure=structure)

    fwhm_mask = create_multi_scale_erosion(
        mask=mask, fits_header=beam_fits_header, scale=scale, scale_is_fwhm=True
    )
    wsclean_mask = create_multi_scale_erosion(
        mask=mask, fits_header=beam_fits_header, scale=scale
    )

    assert np.array_equal(fwhm_mask, expected_mask)
    assert np.sum(fwhm_mask) < np.sum(wsclean_mask)


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
        names = create_snr_mask_from_fits(
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


def test_beam_shape_erode_sorts_scales(
    beam_fits_header: fits.Header, caplog: pytest.LogCaptureFixture
) -> None:
    """wsclean reads the n'th bit as its n'th smallest scale, so unsorted scales
    should be sorted, with a warning, before the bits are assigned"""
    yy, xx = np.mgrid[:256, :256]
    mask = (((xx - 128) ** 2 + (yy - 128) ** 2) <= 40**2).astype(float)

    with caplog.at_level(logging.WARNING, logger="eye-patch"):
        sorted_mask = beam_shape_erode(
            mask=mask, fits_header=beam_fits_header, scales=(0, 4, 16)
        )
    assert not any("ascending order" in record.message for record in caplog.records)

    with caplog.at_level(logging.WARNING, logger="eye-patch"):
        unsorted_mask = beam_shape_erode(
            mask=mask, fits_header=beam_fits_header, scales=(16, 0, 4)
        )
    assert any("ascending order" in record.message for record in caplog.records)
    assert np.array_equal(sorted_mask, unsorted_mask)


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


def _celestial_header(crpix: tuple[float, float]) -> fits.Header:
    """A simple celestial header that only differs in its reference pixel"""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs.wcs.crval = [150.0, -30.0]
    wcs.wcs.cdelt = [-0.001, 0.001]
    wcs.wcs.crpix = list(crpix)
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


def test_create_flood_fill_mask_mac_grows_low_snr_islands() -> None:
    """With the minimum absolute clip the image is not a signal map. Low SNR islands
    should be found using the clip as an increase factor of the same clipping level,
    rather than comparing a significance to the raw pixel values."""
    noise = 1e-4  # An image in Jy/beam rather than a signal map
    yy, xx = np.mgrid[:400, :400]
    diffuse = ((xx - 200) ** 2 + (yy - 200) ** 2) <= 60**2
    image = (
        np.random.default_rng(3).normal(scale=noise, size=(400, 400))
        + diffuse * 2.5 * noise
    )

    without_growth, with_growth = (
        create_flood_fill_mask(
            base_image=image,
            masking_options=MaskingOptions(
                flood_fill=True,
                flood_fill_use_mac=True,
                grow_low_snr_island=grow_low_snr_island,
                grow_low_snr_island_clip=0.5,
            ),
        )
        for grow_low_snr_island in (False, True)
    )

    assert np.mean(without_growth[diffuse]) < 0.1
    assert np.mean(with_growth[diffuse]) > 0.95
    assert np.mean(with_growth[~diffuse]) < 0.02


def test_reverse_negative_flood_fill_is_deprecated() -> None:
    """The old name still works for now, but warns and points to the new name"""
    image = np.random.default_rng(3).normal(size=(100, 100))
    image[40:60, 40:60] += 10.0
    masking_options = MaskingOptions(flood_fill=True)

    with pytest.warns(DeprecationWarning, match="create_flood_fill_mask"):
        deprecated_mask = reverse_negative_flood_fill(
            base_image=image, masking_options=masking_options
        )

    assert np.array_equal(
        deprecated_mask,
        create_flood_fill_mask(base_image=image, masking_options=masking_options),
    )


def test_logger_is_configured(caplog: pytest.LogCaptureFixture) -> None:
    """The eye-patch logger has its own handler so its messages are visible when
    used by other packages (e.g. flint in a dask worker), and is used by masking"""
    assert logger.name == "eye-patch"
    assert logger.level == logging.INFO
    assert any(
        isinstance(handler.formatter, CustomFormatter) for handler in logger.handlers
    )

    with caplog.at_level(logging.INFO, logger="eye-patch"):
        minimum_absolute_clip(image=np.ones(SHAPE), box_size=10)
    assert any(record.name == "eye-patch" for record in caplog.records)


def test_masking_import_configures_logger() -> None:
    """Importing only the masking module, as another package would, should attach
    the eye-patch handler. A fresh interpreter is used as this test module already
    imports the logging module directly."""
    code = (
        "import logging, eye_patch.masking; "
        "eye_patch_logger = logging.getLogger('eye-patch'); "
        "assert eye_patch_logger.level == logging.INFO; "
        "assert eye_patch_logger.handlers"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_fits_masking_mac_without_flood_fill_uses_signal(fits_dir: Path) -> None:
    """The mac is only used with the flood fill, so the base_snr_clip should be
    applied to the signal image (5 everywhere) rather than the raw image (1)"""
    masking_options = MaskingOptions(flood_fill=False, flood_fill_use_mac=True)
    names = create_snr_mask_from_fits(
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
        create_snr_mask_from_fits(
            fits_image_path=fits_dir / "image.fits",
            fits_rms_path=None,
            fits_bkg_path=None,
            masking_options=masking_options,
        )


def test_verify_set_positive_seed_clip_with_nan() -> None:
    """NaNs should not disable the lowering of an unreachable seed clip"""
    signal = np.array([[np.nan, 3.0], [1.0, 2.0]])
    clip = _verify_set_positive_seed_clip(positive_seed_clip=10.0, signal=signal)
    assert np.isclose(clip, 2.7)


@pytest.mark.parametrize("seeds_within_floor", [True, False])
def test_flood_fill_matches_binary_dilation(seeds_within_floor: bool) -> None:
    """The labelling flood fill should match a constrained binary dilation that is
    iterated until convergence, including when seeds sit outside the floor mask"""
    rng = np.random.default_rng(42)
    if seeds_within_floor:
        signal = rng.normal(size=(300, 300))
        seed_mask = signal > 2.5
        floor_mask = signal > 0.3
    else:
        seed_mask = rng.random(size=(300, 300)) > 0.995
        floor_mask = rng.random(size=(300, 300)) > 0.45

    expected = binary_dilation(
        input=seed_mask, mask=floor_mask, iterations=-1, structure=np.ones((3, 3))
    )
    flood_mask = _flood_fill(seed_mask=seed_mask, floor_mask=floor_mask)

    assert np.array_equal(flood_mask, expected)


def test_minimum_absolute_clip_threshold() -> None:
    """A shared threshold should reproduce the minimum absolute clip at any factor"""
    image = np.random.default_rng(42).normal(size=(300, 300))
    threshold = minimum_absolute_clip_threshold(image=image, box_size=30)

    rolling_minimum = minimum_filter(input=image, size=30, mode="wrap")
    for increase_factor in (0.8, 1.5, 4.5):
        expected = image > (increase_factor * np.abs(rolling_minimum))
        assert np.array_equal(image > (threshold * increase_factor), expected)
        assert np.array_equal(
            minimum_absolute_clip(
                image=image, increase_factor=increase_factor, box_size=30
            ),
            expected,
        )


@pytest.mark.parametrize("box_size", [24, 25])
def test_create_boxcar_skew_mask_matches_convolution(box_size: int) -> None:
    """The running mean should match a zero padded convolution with a box of ones,
    including the alignment of even sized boxes"""
    image = np.random.default_rng(42).normal(size=(300, 300)) + 0.2

    result = create_boxcar_skew_mask(image=image, skew_delta=0.2, box_size=box_size)

    expected = fftconvolve(
        (image > 0).astype(float), np.ones((box_size, box_size)), mode="same"
    ) / (box_size**2)
    assert np.allclose(result.positive_pixel_frac, np.clip(expected, 0.0, 1.0))


def test_grow_low_snr_mask_island_sizes() -> None:
    """Only islands of at least the minimum size should be kept"""
    signal = np.random.default_rng(42).normal(size=(300, 300))
    island_size = 40

    low_snr_mask = grow_low_snr_mask(
        signal=signal, grow_low_snr=0.5, grow_low_island_size=island_size
    )

    island_labels, no_islands = label(input=low_snr_mask, structure=np.ones((3, 3)))
    assert no_islands > 0
    island_counts = np.bincount(island_labels.ravel())[1:]
    assert np.all(island_counts >= island_size)
