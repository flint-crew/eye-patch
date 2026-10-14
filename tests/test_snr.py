from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from eye_patch.snr import _create_signal_from_rmsbkg


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
