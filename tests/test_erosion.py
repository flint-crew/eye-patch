from __future__ import annotations

import logging

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from radio_beam import Beam
from scipy.ndimage import binary_erosion
from scipy.signal import fftconvolve

from eye_patch.erosion import (
    beam_shape_erode,
    create_beam_mask_kernel,
    create_multi_scale_erosion,
    fft_binary_erosion,
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


def _disk_centroid(image: np.ndarray) -> tuple[float, float]:
    """Intensity weighted centroid of the positive pixels in an image"""
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
    weights = np.clip(image, 0.0, None)
    return (
        float(np.sum(yy * weights) / np.sum(weights)),
        float(np.sum(xx * weights) / np.sum(weights)),
    )


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
