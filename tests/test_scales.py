from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from eye_patch.cli import cli
from eye_patch.naming import mask_fits_from_scale_mask
from eye_patch.scales import flatten_scale_mask, flatten_scale_mask_fits, get_parser

SHAPE = (100, 100)


def _scale_mask() -> np.ndarray:
    """A per-scale mask with pixels set at one, two and three scales"""
    scale_mask = np.zeros(SHAPE, dtype=np.float32)
    scale_mask[10:20, 10:20] = 1  # The first scale only
    scale_mask[30:40, 30:40] = 2  # The second scale only
    scale_mask[50:60, 50:60] = 3  # The first and second scales
    scale_mask[70:80, 70:80] = 6  # The second and third scales
    return scale_mask


@pytest.fixture
def scale_mask_fits(tmp_path: Path) -> Path:
    path = tmp_path / "image.scalemask.fits"
    fits.writeto(path, _scale_mask(), header=fits.Header({"OBJECT": "test"}))
    return path


def test_flatten_scale_mask() -> None:
    """A pixel set at any scale becomes 1, and every other pixel 0"""
    scale_mask = _scale_mask()
    scale_mask[0, 0] = np.nan

    mask = flatten_scale_mask(scale_mask=scale_mask)

    assert mask.dtype == np.float32
    assert mask.shape == scale_mask.shape
    assert set(np.unique(mask)) == {0.0, 1.0}
    expected = np.isfinite(scale_mask) & (scale_mask != 0)
    assert np.array_equal(mask == 1, expected)
    assert mask[0, 0] == 0, "A non-finite pixel should not be masked"


@pytest.mark.parametrize(
    ("scale_mask_fits", "mask_fits"),
    [
        ("image.scalemask.fits", "image.mask.fits"),
        ("image.fits", "image.mask.fits"),
        ("some/dir/field.scalemask.fits", "some/dir/field.mask.fits"),
        ("some/dir/field.fits", "some/dir/field.mask.fits"),
    ],
)
def test_mask_fits_from_scale_mask(scale_mask_fits: str, mask_fits: str) -> None:
    """A .scalemask.fits ending becomes .mask.fits, and otherwise .fits does"""
    assert mask_fits_from_scale_mask(scale_mask_fits=scale_mask_fits) == Path(mask_fits)


def test_flatten_scale_mask_fits(scale_mask_fits: Path) -> None:
    """The flattened mask is named after the per-scale mask, keeping its header"""
    output_fits = flatten_scale_mask_fits(scale_mask_fits=scale_mask_fits)

    assert output_fits == scale_mask_fits.parent / "image.mask.fits"
    with fits.open(output_fits) as hdul:
        assert hdul[0].header["OBJECT"] == "test"
        assert np.array_equal(hdul[0].data == 1, _scale_mask() != 0)


def test_flatten_scale_mask_fits_to_output(
    scale_mask_fits: Path, tmp_path: Path
) -> None:
    """An output path is used in place of the default name"""
    output_fits = flatten_scale_mask_fits(
        scale_mask_fits=scale_mask_fits, output_fits=tmp_path / "other.fits"
    )

    assert output_fits == tmp_path / "other.fits"
    assert output_fits.exists()
    assert not (tmp_path / "image.mask.fits").exists()


def test_flatten_scale_mask_fits_with_degenerate_axes(tmp_path: Path) -> None:
    """A per-scale mask with degenerate axes keeps its shape"""
    scale_mask_fits = tmp_path / "cube.scalemask.fits"
    fits.writeto(scale_mask_fits, _scale_mask().reshape(1, 1, *SHAPE))

    output_fits = flatten_scale_mask_fits(scale_mask_fits=scale_mask_fits)

    mask = fits.getdata(output_fits)
    assert mask.shape == (1, 1, *SHAPE)
    assert np.array_equal(np.squeeze(mask) == 1, _scale_mask() != 0)


@pytest.mark.parametrize("output", [None, "flat.fits"])
def test_cli_scales_flatten(
    scale_mask_fits: Path, monkeypatch: pytest.MonkeyPatch, output: str | None
) -> None:
    """The eye-patch CLI flattens a per-scale mask, optionally to an output path"""
    argv = ["eye-patch", "scales", "flatten", str(scale_mask_fits)]
    expected = scale_mask_fits.parent / "image.mask.fits"
    if output is not None:
        expected = scale_mask_fits.parent / output
        argv += ["--output-fits", str(expected)]
    monkeypatch.setattr(sys, "argv", argv)

    cli()

    assert expected.exists()
    assert np.array_equal(fits.getdata(expected) == 1, _scale_mask() != 0)


def test_cli_scales_without_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scales sub-command needs an operation"""
    monkeypatch.setattr(sys, "argv", ["eye-patch", "scales"])

    with pytest.raises(SystemExit) as error:
        cli()

    assert error.value.code == 2


def test_scales_parser_is_standalone() -> None:
    """The scales get_parser builds a parser of its own when not given one"""
    args = get_parser().parse_args(
        args=["flatten", "image.scalemask.fits", "--output-fits", "out.fits"]
    )

    assert args.operation == "flatten"
    assert args.scale_mask == Path("image.scalemask.fits")
    assert args.output_fits == Path("out.fits")
    assert not args.overwrite


def test_flatten_scale_mask_fits_will_not_replace_an_existing_output(
    scale_mask_fits: Path, tmp_path: Path
) -> None:
    """An existing output is an error by default, and is left untouched"""
    existing = tmp_path / "image.mask.fits"
    fits.writeto(existing, np.full(SHAPE, 7.0))

    with pytest.raises(FileExistsError, match="already exists"):
        flatten_scale_mask_fits(scale_mask_fits=scale_mask_fits)

    assert np.all(fits.getdata(existing) == 7.0)


def test_flatten_scale_mask_fits_overwrites_with_a_warning(
    scale_mask_fits: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """With overwrite an existing output is replaced, with a warning"""
    existing = tmp_path / "image.mask.fits"
    fits.writeto(existing, np.full(SHAPE, 7.0))

    with caplog.at_level(logging.WARNING, logger="eye-patch"):
        flatten_scale_mask_fits(scale_mask_fits=scale_mask_fits, overwrite=True)

    assert "Overwriting the existing" in caplog.text
    assert np.array_equal(fits.getdata(existing) == 1, _scale_mask() != 0)


@pytest.mark.parametrize("through_link", [False, True])
def test_flatten_scale_mask_fits_will_not_replace_its_input(
    scale_mask_fits: Path, tmp_path: Path, through_link: bool
) -> None:
    """The per-scale mask is never replaced by its flattened mask by default, even
    when the output reaches it through a symbolic link"""
    output_fits = scale_mask_fits
    if through_link:
        output_fits = tmp_path / "link.fits"
        output_fits.symlink_to(scale_mask_fits)

    with pytest.raises(FileExistsError, match="is also an input"):
        flatten_scale_mask_fits(
            scale_mask_fits=scale_mask_fits, output_fits=output_fits
        )

    assert np.array_equal(fits.getdata(scale_mask_fits), _scale_mask())


def test_flatten_scale_mask_fits_replaces_its_input_with_a_warning(
    scale_mask_fits: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """With overwrite the per-scale mask can be replaced, with a warning that its
    scales are lost"""
    with caplog.at_level(logging.WARNING, logger="eye-patch"):
        flatten_scale_mask_fits(
            scale_mask_fits=scale_mask_fits,
            output_fits=scale_mask_fits,
            overwrite=True,
        )

    assert "Replacing the input" in caplog.text
    assert np.array_equal(fits.getdata(scale_mask_fits) == 1, _scale_mask() != 0)


def test_flatten_scale_mask_fits_returns_a_path(
    scale_mask_fits: Path, tmp_path: Path
) -> None:
    """The output is returned as a Path, even when given as a str"""
    output_fits = flatten_scale_mask_fits(
        scale_mask_fits=str(scale_mask_fits), output_fits=str(tmp_path / "out.fits")
    )

    assert isinstance(output_fits, Path)
    assert output_fits == tmp_path / "out.fits"


def test_cli_scales_flatten_overwrite(
    scale_mask_fits: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --overwrite an existing output is an error, and with it the output
    is replaced"""
    existing = scale_mask_fits.parent / "image.mask.fits"
    fits.writeto(existing, np.full(SHAPE, 7.0))
    argv = ["eye-patch", "scales", "flatten", str(scale_mask_fits)]

    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileExistsError):
        cli()
    assert np.all(fits.getdata(existing) == 7.0)

    monkeypatch.setattr(sys, "argv", [*argv, "--overwrite"])
    cli()
    assert np.array_equal(fits.getdata(existing) == 1, _scale_mask() != 0)
