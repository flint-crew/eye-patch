from __future__ import annotations

import logging
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np

from eye_patch.logging import CustomFormatter, logger
from eye_patch.mac import (
    minimum_absolute_clip,
)

if TYPE_CHECKING:
    import pytest

SHAPE = (100, 100)


def test_logger_is_configured(caplog: pytest.LogCaptureFixture) -> None:
    """The eye-patch logger has its own handler so its messages are visible when
    used by other packages (e.g. flint in a dask worker), and is used by masking"""
    assert logger.name == "eye-patch"
    assert logger.level == logging.INFO
    assert any(
        isinstance(handler.formatter, CustomFormatter) for handler in logger.handlers
    )

    with caplog.at_level(logging.INFO, logger="eye-patch"):
        minimum_absolute_clip(image=np.ones(SHAPE), box_size=10)
    assert any(record.name == "eye-patch" for record in caplog.records)


def test_masking_import_configures_logger() -> None:
    """Importing only the masking module, as another package would, should attach
    the eye-patch handler. A fresh interpreter is used as this test module already
    imports the logging module directly."""
    code = (
        "import logging, eye_patch.masking; "
        "eye_patch_logger = logging.getLogger('eye-patch'); "
        "assert eye_patch_logger.level == logging.INFO; "
        "assert eye_patch_logger.handlers"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
