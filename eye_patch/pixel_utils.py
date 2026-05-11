from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.io import fits

if TYPE_CHECKING:
    from pathlib import Path

    from radio_beam import Beam


@dataclass(frozen=True)
class BeamShape:
    """A simple container to represent a fitted 2D gaussian,
    intended for the main lobe of the synthesised beam. This
    class has been defined to avoid issues with the serialisation
    of astropy.units, which can cause strange and wonderful
    errors when being sent over the wire to workers."""

    bmaj_arcsec: float
    """The size of the major-axis of the beam, in arcseconds."""
    bmin_arcsec: float
    """The size of the minor-axis of the beam, in arcseconds."""
    bpa_deg: float
    """Rotation of the beam, in degrees."""

    @classmethod
    def from_radio_beam(cls, radio_beam: Beam) -> BeamShape:
        """A helper function to convert a radio_beam.Beam into a
        BeamShape. This is prinicpally intended to be used when
        there is a need to exchange a Beam between processes
        that would need to serialise the object.

        Args:
            radio_beam (Beam): The Beam to convert to normalised and known units

        Returns:
            BeamShape: The normalised container without astropy units.
        """
        return cls(
            bmaj_arcsec=radio_beam.major.to(u.arcsecond).value,
            bmin_arcsec=radio_beam.minor.to(u.arcsecond).value,
            bpa_deg=radio_beam.pa.to(u.degree).value,
        )


def get_beam_shape(fits_path: Path) -> BeamShape | None:
    """Construct and return a beam shape from the fields in a FITS image

    Args:
        fits_path (Path): FITS image to extract the beam information from

    Returns:
        Optional[BeamShape]: Shape of the beam stored in the FITS image. None is returned if the beam is not found.
    """

    header = fits.getheader(filename=fits_path)

    if not all(key in header for key in ("BMAJ", "BMIN", "BPA")):
        return None

    return BeamShape(
        bmaj_arcsec=header["BMAJ"] * 3600,
        bmin_arcsec=header["BMIN"] * 3600,
        bpa_deg=header["BPA"],
    )


def get_pixels_per_beam(fits_path: Path) -> float | None:
    """Given a image with beam information, return the number of pixels
    per beam. The beam is taken from the FITS header. This is evaluated
    for pixels at the reference pixel position.

    Args:
        fits_path (Path): FITS image to consideer

    Returns:
        float | None: Number of pixels per beam. If beam is not in header then None is returned.
    """

    beam_shape = get_beam_shape(fits_path=fits_path)

    if beam_shape is None:
        return None

    header = fits.getheader(filename=fits_path)

    pixel_ra = np.abs(header["CDELT1"] * 3600)
    pixel_dec = np.abs(header["CDELT2"] * 3600)

    assert isinstance(beam_shape, BeamShape), (
        f"Expected type of Beamshape, but have {type(beam_shape)}"
    )
    beam_area = beam_shape.bmaj_arcsec * beam_shape.bmin_arcsec * np.pi
    pixel_area = pixel_ra * pixel_dec

    return float(beam_area / pixel_area)
