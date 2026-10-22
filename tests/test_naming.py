from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from eye_patch.naming import check_output_fits

if TYPE_CHECKING:
    from pathlib import Path


def test_check_output_fits_new_output(tmp_path: Path) -> None:
    """An output that does not exist may always be written"""
    check_output_fits(output_fits=tmp_path / "new.fits", overwrite=False)


def test_check_output_fits_existing_output(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An existing output is an error without overwrite, and a warning with it"""
    output_fits = tmp_path / "out.fits"
    output_fits.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exists"):
        check_output_fits(output_fits=output_fits, overwrite=False)

    with caplog.at_level(logging.WARNING, logger="eye-patch"):
        check_output_fits(output_fits=output_fits, overwrite=True)
    assert "Overwriting the existing" in caplog.text


@pytest.mark.parametrize("through_link", [False, True])
def test_check_output_fits_output_is_an_input(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, through_link: bool
) -> None:
    """An output that is also an input is reported as such, even when reached
    through a symbolic link, and inputs of None are skipped"""
    input_fits = tmp_path / "input.fits"
    input_fits.write_bytes(b"input")
    output_fits = input_fits
    if through_link:
        output_fits = tmp_path / "link.fits"
        output_fits.symlink_to(input_fits)
    inputs = (None, tmp_path / "missing.fits", input_fits)

    with pytest.raises(FileExistsError, match="is also an input"):
        check_output_fits(output_fits=output_fits, overwrite=False, input_fits=inputs)

    with caplog.at_level(logging.WARNING, logger="eye-patch"):
        check_output_fits(output_fits=output_fits, overwrite=True, input_fits=inputs)
    assert "Replacing the input" in caplog.text
