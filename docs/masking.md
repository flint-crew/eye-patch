# Creating Clean Masks

The imaging performance of CLEAN and its derivatives can be improved through the
construction of reliable clean mask regions. These regions restrict the set of
pixels that are involved during the peak finding stage of CLEAN. Here the aim of
the game is to optimally select pixels that contain genuine emission. Issues
such as clean bias, sub-optimal self-calibration and clean divergence can be
minimised or outright eliminated provided a clean mask has been reliably
constructed.

`eye-patch` provides functionality to construct such clean masks, where the
output is a pixel-wise mask of arbitrary shapes that either allow or deny
cleaning to occur at certain pixel locations. These are intended to be evaluated
against a restored FITS image. Pixels in the output clean mask are either `0`
(cleaning is not allowed here) or `1` (cleaning is allowed here), apart from the
per-scale masks described below.

## Available statistics

`eye-patch` currently supports two statistics to identify pixels of significance
that should be cleaned.

### Signal-to-noise (SNR)

The obvious method is based on signal-to-noise (SNR). After constructing
background ({math}`\mu`) and noise ({math}`\sigma`) measures across an image, a
direction-dependent SNR ({math}`\mathrm{SNR}`) across the image
({math}`\mathrm{img}`) can be expressed as:

{math}`\mathrm{SNR}(x,y) = \frac{\left(\mathrm{img}-\mu\right)}{\sigma}.`

In the simplest case, constant values may be set across the extent of the image,
with {math}`\mu=0` and {math}`\sigma=\mathrm{SD}(\mathrm{img})`. These can be
replaced with more sophisticated schemes that compute position-dependent metrics
that also incorporate iterative outlier clipping (e.g.
[Background and Noise Estimator](https://github.com/PaulHancock/Aegean)) or
robust statistics (e.g.
[Selavy](https://www.atnf.csiro.au/computing/software/askapsoft/sdp/docs/current/analysis/selavy.html)).

At their heart, these SNR-based processes assume Gaussian-distributed, zero-mean
noise, where pixel intensities that are unlikely to occur by chance are assumed
to be genuine emission. That is to say, pixels that are {math}`\gg5\sigma` are
likely to be real and should be included in a mask for cleaning. Without the
flood fill (see below), the pixels with an SNR above `MaskingOptions.base_clip`
form the mask.

### Minimum absolute clip

SNR-based measures have a clear statistical foundation when identifying bright
pixels. However, there are situations when such a metric could be perturbed or
otherwise corrupted, including:

- Calibration errors that produce imaging artefacts,
- Deconvolution errors that accumulate over minor/major iterations,
- Mishandling of the w-term so that the 2D Fourier transform approximation
  breaks down,
- Extended diffuse structures that represent a large fraction of the region used
  to calculate the local {math}`\mu` or {math}`\sigma` quantities,
- Combinations of the above, or
- Gremlins lurking around in the data.

Ultimately, if the regions being considered in the derivation of the noise do
not appear Gaussian-like, the robustness of methods to calculate the local noise
level is suspect. Additionally, it is unclear whether an accurate noise
estimation is even an appropriate basis for such regions. For instance,
artefacts around a bright source produced by phase error could be
{math}`\gg\sigma` if the genuine source is sufficiently bright.

We introduce the Minimum Absolute Clip ({math}`\mathrm{MAC}`) as an alternative
metric. It assumes:

1. approximately zero-mean Gaussian noise,
2. the sky brightness is positive definite, and
3. should there be a significantly bright negative component, there will be a
   positive one of comparable brightness nearby.

These assumptions are fair for Stokes-I images. A masking function can therefore
be constructed, where:

{math}`\mathrm{MAC}(x, y) = \mathrm{img} > P \times |\mathrm{RollingMinimum}\left(\mathrm{img}, \mathrm{BoxSize}(\mathrm{img}, x, y)\right)|`

where {math}`\mathrm{RollingMinimum}` is a minimum boxcar filter whose size, in
pixels, is returned by {math}`\mathrm{BoxSize}`, which could be tuned in a
position-dependent way. The absolute value of the
{math}`\mathrm{RollingMinimum}` function is multiplied by an increase factor
{math}`P` to raise the minimum positive threshold. The {math}`\mathrm{MAC}` is a
simple and efficient statistic to compute.

The choice of an appropriate boxcar size is an important consideration. A larger
size enforces more conservative behaviour from the {math}`\mathrm{MAC}` process.
Should too small a region be supplied, then large multi-scale features are
typically less reliably assessed. Extensions can be made to detect when the
rolling box filter size is too small (e.g. by an imbalance of positive to
negative pixels). A basic adaptive boxcar size is implemented via the
`MaskingOptions.mac_adaptive_max_depth`,
`MaskingOptions.mac_adaptive_step_factor` and
`MaskingOptions.mac_adaptive_skew_delta` parameters.

<!-- The figures are made with docs/scripts/adaptive_mac_figure.py -->

```{figure} _static/adaptive_mac_light.png
:alt: A cut through an image, where a fixed box gives a clip level that rises far above a wide, bright source and leaves its middle unmasked, while the adaptive box keeps the clip level at the noise and masks the whole source.
:figclass: only-light
:width: 100%

The top panel is a cut through an image of unit noise (grey), with a compact
source and a bright source that is wider than the default 75-pixel box. The blue
and orange lines are the clip levels of the {math}`\mathrm{MAC}`, with an
increase factor of 4, from a fixed box and from the adaptive box
(`mac_adaptive_max_depth=3`). A pixel is included in the clean mask wherever the
image is above the clip level, and the bottom panel shows the resulting masks along the cut, where
the bars are True. With a fixed box, the rolling minimum in the middle of the
wide source only sees the source, so the clip level rises above it and the
middle of the source is missed, leaving a hole. The adaptive box doubles in size
wherever positive pixels dominate it, so its clip level stays at the noise and
the whole source is masked. Away from the wide source the two clip levels are
the same, and their lines overlap.
```

```{figure} _static/adaptive_mac_dark.png
:alt: A cut through an image, where a fixed box gives a clip level that rises far above a wide, bright source and leaves its middle unmasked, while the adaptive box keeps the clip level at the noise and masks the whole source.
:figclass: only-dark
:width: 100%

The top panel is a cut through an image of unit noise (grey), with a compact
source and a bright source that is wider than the default 75-pixel box. The blue
and orange lines are the clip levels of the {math}`\mathrm{MAC}`, with an
increase factor of 4, from a fixed box and from the adaptive box
(`mac_adaptive_max_depth=3`). A pixel is included in the clean mask wherever the
image is above the clip level, and the bottom panel shows the resulting masks along the cut, where
the bars are True. With a fixed box, the rolling minimum in the middle of the
wide source only sees the source, so the clip level rises above it and the
middle of the source is missed, leaving a hole. The adaptive box doubles in size
wherever positive pixels dominate it, so its clip level stays at the noise and
the whole source is masked. Away from the wide source the two clip levels are
the same, and their lines overlap.
```

The {math}`\mathrm{MAC}` is selected with `MaskingOptions.use_mac`, and may be
used with or without the flood fill (see below). Without the flood fill, the
increase factor {math}`P` is `MaskingOptions.base_clip`, and with it, the seed
and flood clips take its place. As the {math}`\mathrm{MAC}` operates on the
image itself, no background or noise images are needed.

## Reverse flood filling

Cleaning sources with features that are both faint and diffuse is difficult. On
a per-pixel basis it is often a reality that a minimum signal-to-noise threshold
is not met. Should the cleaning thresholds (set in either absolute
{math}`\mathrm{Jy}/\mathrm{beam}` units or in terms of {math}`\sigma` units) be
too high, this diffuse emission is never cleaned. In `eye-patch` we include a
reverse flood fill procedure that grows islands of pixels above an initial
criterion out to adjacent pixels that meet a secondary, lower level. This
process is the initial step of most source finders.

There are two steps to this process:

1. _Initial significant island detection_: The statistic of choice (see above)
   is computed, and all pixels above the threshold are marked as pixels to
   clean. This threshold is set via `MaskingOptions.seed_clip`.
2. _Growing the initial set of islands_: Islands that are identified as
   significant go through a dilation process, which is restricted to the pixels
   above a lower threshold. This lower clip is set via
   `MaskingOptions.flood_clip`.

The flood fill procedure is 'opt in', and is activated via
`MaskingOptions.flood_fill`. Either statistic may be used to seed and grow
islands. With the {math}`\mathrm{MAC}`, the seed and flood clips are increase
factors {math}`P` of the rolling minimum, rather than SNR levels.

## Islands of faint and diffuse emission

Islands of faint and diffuse emission that would otherwise be missed may also be
added via `MaskingOptions.grow_low_snr_island`, with or without the flood fill.
Pixels above `MaskingOptions.grow_low_snr_island_clip` are joined together, and
islands made up of at least `MaskingOptions.grow_low_snr_island_min_pixels`
pixels are added to the mask. When the {math}`\mathrm{MAC}` statistic is used,
the clip is an increase factor of the same clipping level as the other clips.
For a 75-pixel box over noise, this clipping level is around {math}`3.7\sigma`,
so a factor near `0.5` is comparable to a {math}`1.75\sigma` clip of a signal
map.

This option is 'opt in', and complements rather than replaces the flood fill or
`MaskingOptions.base_clip`. Its islands are formed separately from the rest of
the mask, using only `MaskingOptions.grow_low_snr_island_clip` and
`MaskingOptions.grow_low_snr_island_min_pixels` rather than the seed and flood
clips, and are then added to the mask. As it only ever adds pixels, it can go
awry. Anything it picks up in error, such as a large region of noise, artefacts
or poorly subtracted background above its clip, stays in the mask, as nothing
later removes such an island. The beam erosion (see below), should it be used,
only trims the edges of islands. Care should be taken with a low clip or a small
minimum island size, as both make such errors more likely.

<!-- TODO: Need to include some other image here -->

## Eroding output islands

Output binary clean masks are naturally at the resolution of the input image (at
least in `eye-patch`). All pixels that reside in an island are candidate pixels
where cleaning is allowed to occur. Consider an unresolved source that is
perfectly aligned to the discrete pixel grid. In principle, its clean component
should be confined to a single pixel. However, if the source is sufficiently
bright, the resulting mask is made up of many pixels (i.e. the shape of the
restoring beam). Should deep cleaning be employed, it is possible that small
perturbations from the noise underlying a source would also be cleaned.

A binary erosion process with the erosion structure set to the shape of the
restoring beam at a particular power level can be used to 'contract' all pixel
masks. The output should better reflect where clean components ought to be
placed. See the `MaskingOptions.beam_shape_erode` and
`MaskingOptions.beam_shape_erode_minimum_response` options to activate and
control this process. The minimum response should be set between 0 and 1
(exclusive), where numbers closer to 0 represent a larger binary erosion
structure shape. That is to say, islands need to be _larger_ for any pixels to
remain as `MaskingOptions.beam_shape_erode_minimum_response` approaches 0.

### Multi-scale beam erosion

Basic routines to enable per-scale beam shape erosion are provided via
`MaskingOptions.multiscale_scales`, together with
`MaskingOptions.beam_shape_erode`. For each specified scale the restoring beam
(the base shape in this erosion) is convolved with the circular Gaussian that
wsclean uses for that scale. The resulting beam is then evaluated in the basic
process outlined above, eroding the clean islands at the native resolution to
the clean islands at the specified scale.

The scales are wsclean multi-scale scales in pixels, and are not the FWHM of the
Gaussian. wsclean (via radler) forms the Gaussian of a scale with
{math}`\sigma = 3/16` of the scale, which is a FWHM of about `0.44` of the
scale. A scale of 16 pixels is therefore a Gaussian with a FWHM of about 7
pixels. These masks are intended for wsclean's Gaussian scales
(`-multiscale-shape gaussian`).

```{note}
Earlier versions of `eye-patch` used each scale as the FWHM of the Gaussian.
This made the erosion structure of a scale about 2.3 times wider than the
wsclean Gaussian of the same scale, so larger scales were eroded more than
intended. Masks at larger scales are now larger than before, and settings tuned
against the earlier behaviour may need to be revisited. The example image below
was made with the earlier behaviour.

To compare against the earlier behaviour, set
`MaskingOptions.beam_shape_erode_scales_as_fwhm` (or
`--beam-shape-erode-scales-as-fwhm` on the CLI). Only the width of each scale is
restored. Other fixes, such as centring the erosion structure on a pixel, still
apply.
```

The clean mask of each scale is stored as a 'bitmapped' value. As the masks are
floats, a pixel within the mask of the `n`'th scale has `2**n` added to it.
Scales are first sorted into ascending order, as wsclean reads the `n`'th bit as
its `n`'th smallest scale.

When more than one scale is specified, this bitmapped mask is written to
`<image>.scalemask.fits`. With a single scale, or without erosion, the mask is
written to `<image>.mask.fits`. Should the image header be missing the beam
information, no erosion is performed, and the mask is also written to
`<image>.mask.fits`.

This is intended to provide a mechanism to ensure cleaning at larger scales
occurs in well-defined regions. An example of the erosion process for
`0 1 2 4 8 16 32 64 128 256 512 1024 2056` is shown below. Point source islands
are still in the mask. As they are only present at the first scale, they are
stored as `2**0 = 1`.

```{image} _static/multi_scale_erode.jpg
:alt: Example of multi-scale beam erode
:width: 100%
```

## `--convolve-first` option

Historically, for extragalactic fields the reverse flood fill process described
above, using a minimum absolute clip as the threshold statistic, produced
reliable clean masks. A subsequent binary erosion process to attempt to
characterise per-scale clean masks produced meaningful results. For regions like
the Galactic Plane, however, the reverse flood fill process would often miss
diffuse emission that was both extended and below the flood fill clip level.
Reducing this lower level clip would not consistently capture such faint
extended structures, and pixels boosted by noise would often leak into the clean
mask from legitimate islands (flying spaghetti monsters).

The `--convolve-first` option activates a routine where the base image is first
put through an `open` filter before being convolved to some desired scale, using
the same Gaussian that wsclean uses for that scale. Significance thresholding
operations are then applied to this filtered image. Importantly, the `open`
filter attempts to isolate small structures from larger ones. This is
implemented through the combined usage of a minimum filter followed by a maximum
filter (could be considered a high-pass filter). The per-scale clean masks are
always written to `<image>.scalemask.fits`, using the same bitmapped
representation as the multi-scale beam erosion.

```{raw} html
<video width="100%" controls>
  <source src="_static/open_filter.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
```

Should this option be activated, be aware that the seed and flood clipping
thresholds change, and it should be possible to use lower limits. For example,
`seed_clip=1.02` and `flood_clip=0.15` may be used, as these are increase
factors of the minimum absolute clip, which this option always uses. At the
native scale (a scale of `0`), the flood clip is raised to at least `0.8`.

## Accessing via the CLI

The masking utility may be accessed via the `eye-patch` CLI entry point:

```{argparse}
:ref: eye_patch.cli.get_parser
:prog: eye_patch_masking
```

## The `MaskingOptions` class

Embedded below is the `MaskingOptions` class used to construct clean masks.
Input values are validated by `pydantic` to ensure they are appropriately typed.

```{literalinclude} ../eye_patch/options.py
:pyobject: MaskingOptions
```

## Changes from v1.0.1 and earlier versions

### Renamed options

Several `MaskingOptions` attributes were renamed so that each name describes
what the option does, rather than where it was first used. The table below maps
each earlier name to its new name.

| Earlier name                              | New name                         |
| ----------------------------------------- | -------------------------------- |
| `base_snr_clip`                           | `base_clip`                      |
| `flood_fill_positive_seed_clip`           | `seed_clip`                      |
| `flood_fill_positive_flood_clip`          | `flood_clip`                     |
| `flood_fill_use_mac`                      | `use_mac`                        |
| `flood_fill_use_mac_box_size`             | `mac_box_size`                   |
| `flood_fill_use_mac_adaptive_step_factor` | `mac_adaptive_step_factor`       |
| `flood_fill_use_mac_adaptive_skew_delta`  | `mac_adaptive_skew_delta`        |
| `flood_fill_use_mac_adaptive_max_depth`   | `mac_adaptive_max_depth`         |
| `beam_shape_erode_scales`                 | `multiscale_scales`              |
| `grow_low_snr_island_size`                | `grow_low_snr_island_min_pixels` |

The CLI flags follow the same mapping, with dashes in place of underscores. For
example, `--flood-fill-positive-seed-clip` is now `--seed-clip`.

Earlier names are no longer accepted. An earlier name, whether in a strategy
file or passed to `MaskingOptions`, raises a validation error rather than being
silently ignored, so a stale setting is caught when it is loaded. The masking
options of [Flint](https://github.com/flint-crew/flint) use the earlier names,
so this table also maps a Flint strategy file onto `eye-patch`.

### Changes in behaviour

- **The {math}`\mathrm{MAC}` no longer needs the flood fill.** `use_mac`
  (formerly `flood_fill_use_mac`) applies with or without `flood_fill`. Without
  the flood fill, the mask is the pixels above `base_clip` times the
  {math}`\mathrm{MAC}` clipping level. Previously `flood_fill_use_mac` was
  ignored unless `flood_fill` was set, and `base_snr_clip` was applied to the
  signal image instead. As the {math}`\mathrm{MAC}` operates on the image
  itself, no signal image is formed, so the RMS and background images are not
  needed.
- **Islands of faint and diffuse emission no longer need the flood fill.**
  `grow_low_snr_island` adds its islands to the mask whether or not `flood_fill`
  is set. Previously it was silently ignored without the flood fill. It remains
  opt in, and complements rather than replaces the flood fill or `base_clip`.
  Its islands are formed separately, from `grow_low_snr_island_clip` and
  `grow_low_snr_island_min_pixels` rather than the seed and flood clips, and are
  then added to the mask. As it only ever adds pixels, it can go awry: anything
  it picks up in error, such as a large region of noise, artefacts or poorly
  subtracted background above its clip, stays in the mask. Nothing later removes
  such an island, as the beam erosion, should it be used, only trims the edges
  of islands.
- **`base_clip` follows the statistic.** With `use_mac`, it is an increase
  factor of the {math}`\mathrm{MAC}` clipping level, as `seed_clip` and
  `flood_clip` already were.
- **Masks made without the flood fill are written as 32-bit floats**, rather
  than 64-bit floats. Their values are unchanged.
- **Masks extracted from a mosaic take the shape of their template image.**
  Should the template carry degenerate frequency and Stokes axes, as images
  written by wsclean do, the extracted masks now carry them too, rather than
  always being two-dimensional. Their values are unchanged.

Settings that set `flood_fill_use_mac` or `grow_low_snr_island` without
`flood_fill` now produce a different mask, and should be revisited. The
`convolve_first` option is unaffected.
