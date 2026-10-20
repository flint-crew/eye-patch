"""Fixtures shared across the test modules"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

SHAPE = (100, 100)


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
