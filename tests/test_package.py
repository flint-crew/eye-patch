from __future__ import annotations

import importlib.metadata

import eye_patch as m


def test_version():
    assert importlib.metadata.version("eye_patch") == m.__version__
