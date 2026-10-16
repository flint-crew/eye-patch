from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from capn_crunch import create_options_from_parser

from eye_patch import extract, masking
from eye_patch.cli import cli, get_parser
from eye_patch.options import MaskingOptions

if TYPE_CHECKING:
    import pytest


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
            "--seed-clip",
            "10",
            "--flood-clip",
            "1.",
            "--use-mac",
            "--mac-box-size",
            "100",
        ]
    )
    masking_options = create_options_from_parser(
        parser_namespace=args, options_class=MaskingOptions
    )
    assert isinstance(masking_options, MaskingOptions)
    assert masking_options.flood_fill
    assert masking_options.use_mac
    assert masking_options.seed_clip == 10.0
    assert masking_options.flood_clip == 1.0


def test_arg_parser_cli_and_masking_options_for_multiscale():
    """See if the CLI parser is constructed with correct set of
    options which are properly converted to a MaskingOptions object. Here
    want to make sure that the multiscale erosion are picked up"""
    parser = get_parser()
    args = parser.parse_args(
        args=[
            "mask",
            "img",
            "--multiscale-scales",
            "0",
            "1",
            "2",
            "4",
            "--rms-fits",
            "rms",
            "--bkg-fits",
            "bkg",
            "--flood-fill",
            "--seed-clip",
            "10",
            "--flood-clip",
            "1.",
            "--use-mac",
            "--mac-box-size",
            "100",
        ]
    )
    masking_options = create_options_from_parser(
        parser_namespace=args, options_class=MaskingOptions
    )
    assert isinstance(masking_options, MaskingOptions)
    assert isinstance(masking_options.multiscale_scales, tuple)
    assert masking_options.multiscale_scales == (0, 1, 2, 4)
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


def test_sub_command_parsers_are_standalone() -> None:
    """Each sub-command's get_parser configures a sub-parser of the eye-patch CLI,
    but should also build a parser of its own when not given one"""
    mask_args = masking.get_parser().parse_args(args=["img", "--flood-fill"])
    assert mask_args.image == Path("img")
    assert mask_args.flood_fill

    extract_args = extract.get_parser().parse_args(args=["beam.fits", "mosaic.fits"])
    assert extract_args.beam_image == Path("beam.fits")
    assert extract_args.mosaic_image == Path("mosaic.fits")


def test_cli_without_mode_prints_help(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Without a sub-command the CLI logs an error and prints its help"""
    monkeypatch.setattr(sys, "argv", ["eye-patch"])
    with caplog.at_level(logging.ERROR, logger="eye-patch"):
        cli()

    assert "Supplied mode None is not known" in caplog.text
    assert "{mask,extractmask}" in capsys.readouterr().out
