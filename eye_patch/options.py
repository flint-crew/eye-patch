"""Options that control the creation of clean masks"""

from __future__ import annotations

from capn_crunch import BaseOptions


class MaskingOptions(BaseOptions):
    """Contains options for the creation of clean masks from some subject
    image. Clipping levels specified are in units of RMS (or sigma). They
    are NOT in absolute units.
    """

    base_clip: float = 4
    """A base clipping level to be used should other options not be activated"""
    flood_fill: bool = False
    """Whether to attempt to flood fill when constructing a mask. This should be `True` for ``grow_low_snr_island`` to have an effect."""
    seed_clip: float = 4.5
    """The clipping level to seed islands that will be grown to lower signal metric"""
    flood_clip: float = 1.5
    """Clipping level used to grow seeded islands down to"""
    use_mac: bool = False
    """If True, the clipping levels (including ``grow_low_snr_island_clip``) are used as the `increase_factor` when using a minimum absolute clip"""
    mac_box_size: int = 75
    """The size of the mac box size should mac be used"""
    mac_adaptive_step_factor: float = 2.0
    """Stepping size used to increase box by should adaptive detect poor boxcar statistics"""
    mac_adaptive_skew_delta: float = 0.2
    """A box is consider too small for a pixel if the fractional proportion of positive pixels is larger than the deviation away of (0.5 + frac). This threshold is therefore 0 to 0.5"""
    mac_adaptive_max_depth: int | None = None
    """Determines the number of adaptive boxcar scales to use when constructing seed mask. If None no adaptive boxcar sizes"""
    grow_low_snr_island: bool = False
    """Whether to attempt to grow a mask to capture islands of low SNR (e.g. diffuse emission)"""
    grow_low_snr_island_clip: float = 1.75
    """The minimum significance levels of pixels to be to seed low SNR islands for consideration. When ``use_mac`` is True this is the `increase_factor` of the minimum absolute clip instead"""
    grow_low_snr_island_min_pixels: int = 768
    """The number of pixels an island has to be for it to be accepted"""
    beam_shape_erode: bool = False
    """Erode the mask using the shape of the restoring beam"""
    beam_shape_erode_minimum_response: float = 0.6
    """The minimum response of the beam that is used to form the erode structure shape"""
    multiscale_scales: tuple[int, ...] | None = None
    """The wsclean multi-scale scales, in pixels, to perform a binary erosion with. Each is converted to the Gaussian that wsclean uses for it (a FWHM of about 0.44 of the scale). Output scales are encoded as a bitmapped value (e.g. n'th scale is n'th bit). Scales are sorted into ascending order to match wsclean"""
    beam_shape_erode_scales_as_fwhm: bool = False
    """Use each of the ``multiscale_scales`` as the FWHM of the Gaussian, rather than the Gaussian wsclean uses for it. This was the behaviour of earlier versions, and is about 2.3 times wider than the wsclean Gaussian of a scale."""
    convolve_first: bool = False
    """Attempt to construct mask across scales by first convolving the input image by a scale kernel, then run the island construction stage"""
