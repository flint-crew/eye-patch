"""The ``eye-patch`` command line interface. Each sub-command is configured by the
``get_parser`` of its module, which is passed the sub-parser to add arguments to."""

from __future__ import annotations

from argparse import ArgumentParser

from eye_patch import extract, masking, scales
from eye_patch.logging import logger


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Simple utility functions to create masks from FITS images. "
    )

    subparser = parser.add_subparsers(
        dest="mode", help="Operation mode of flint_bandpass"
    )

    masking.get_parser(
        parser=subparser.add_parser(
            "mask",
            help="Create a mask for an image, potentially using its RMS and BKG images (e.g. outputs from BANE). Output FITS image will default to the image with a mask suffix.",
        )
    )

    extract.get_parser(
        parser=subparser.add_parser(
            "extractmask",
            help="Extract a beam FITS masked region from a larger FITS mask mosaic image. ",
        )
    )

    scales.get_parser(
        parser=subparser.add_parser(
            "scales",
            help="Operations on a per-scale mask (.scalemask.fits), e.g. flattening it into a single mask.",
        )
    )

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "mask":
        masking.cli(args=args)
    elif args.mode == "extractmask":
        extract.cli(args=args)
    elif args.mode == "scales":
        scales.cli(args=args)
    else:
        logger.error(f"Supplied mode {args.mode} is not known. ")
        parser.print_help()
