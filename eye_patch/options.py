"""Options that control the creation of clean masks"""

from __future__ import annotations

from capn_crunch import BaseOptions


class MaskingOptions(BaseOptions):
    """Contains options for the creation of clean masks from some subject
    image. Clipping levels are relative, and are NOT in absolute units. With the
    signal-to-noise statistic they are in units of RMS (or sigma). With the minimum
    absolute clip, which is selected by ``use_mac`` and is always used by
    ``convolve_first``, they are instead increase factors of its clipping level.
    """

    base_clip: float = 4
    """The clipping level of each pixel when the flood fill is not used. A signal-to-noise level, or an increase factor of the minimum absolute clip should ``use_mac`` be set. Not used by ``convolve_first``"""
    flood_fill: bool = False
    """Grow islands seeded above ``seed_clip`` out to the connected pixels above ``flood_clip``, rather than clipping each pixel at ``base_clip``. Not used by ``convolve_first``, which always flood fills"""
    seed_clip: float = 4.5
    """The clipping level of the pixels that seed islands for the flood fill. A signal-to-noise level, or an increase factor of the minimum absolute clip should ``use_mac`` be set. Also used by ``convolve_first``, as an increase factor"""
    flood_clip: float = 1.5
    """The clipping level that seeded islands are grown down to by the flood fill. A signal-to-noise level, or an increase factor of the minimum absolute clip should ``use_mac`` be set. Also used by ``convolve_first``, as an increase factor that is raised to at least 0.8 at a scale of 0"""
    use_mac: bool = False
    """Clip the image with the minimum absolute clip rather than its signal-to-noise, with or without the flood fill. ``base_clip``, ``seed_clip``, ``flood_clip`` and ``grow_low_snr_island_clip`` are then increase factors of its clipping level. As the image itself is clipped no RMS or background images are needed. Not used by ``convolve_first``, which always uses the minimum absolute clip"""
    mac_box_size: int = 75
    """The size, in pixels, of the boxcar of the rolling minimum used by the minimum absolute clip. ``convolve_first`` adds each scale to it"""
    mac_adaptive_step_factor: float = 2.0
    """The factor the boxcar is enlarged by where the adaptive minimum absolute clip detects poor boxcar statistics"""
    mac_adaptive_skew_delta: float = 0.2
    """A boxcar is considered too small for a pixel when the fraction of positive pixels within it is larger than 0.5 plus this delta, which is therefore between 0 and 0.5"""
    mac_adaptive_max_depth: int | None = None
    """The maximum number of times the adaptive minimum absolute clip may enlarge the boxcar. If None the boxcar is not adapted"""
    grow_low_snr_island: bool = False
    """Add large islands of faint and diffuse emission, formed from the pixels above ``grow_low_snr_island_clip``, to the mask, with or without the flood fill. Not used by ``convolve_first``"""
    grow_low_snr_island_clip: float = 1.75
    """The clipping level of the pixels that form islands of faint and diffuse emission. A signal-to-noise level, or an increase factor of the minimum absolute clip should ``use_mac`` be set"""
    grow_low_snr_island_min_pixels: int = 768
    """The minimum number of pixels an island of faint and diffuse emission needs to be added to the mask"""
    beam_shape_erode: bool = False
    """Erode the mask using the shape of the restoring beam. Not used by ``convolve_first``, which always erodes"""
    beam_shape_erode_minimum_response: float = 0.6
    """The minimum response of the beam that is used to form the erode structure shape"""
    multiscale_scales: tuple[int, ...] | None = None
    """The wsclean multi-scale scales, in pixels. Each is converted to the Gaussian that wsclean uses for it (a FWHM of about 0.44 of the scale). With ``beam_shape_erode`` the mask is eroded at each scale, and ``convolve_first`` convolves the image to each scale. Output scales are encoded as a bitmapped value (e.g. n'th scale is n'th bit). Scales are sorted into ascending order to match wsclean"""
    beam_shape_erode_scales_as_fwhm: bool = False
    """Use each of the ``multiscale_scales`` as the FWHM of the Gaussian, rather than the Gaussian wsclean uses for it. This was the behaviour of earlier versions, and is about 2.3 times wider than the wsclean Gaussian of a scale."""
    convolve_first: bool = False
    """Construct the mask of each scale by first convolving the image to the scale, then flood filling it with the minimum absolute clip and eroding it by the restoring beam. ``base_clip``, ``flood_fill``, ``use_mac``, ``grow_low_snr_island`` and ``beam_shape_erode`` are not used"""
