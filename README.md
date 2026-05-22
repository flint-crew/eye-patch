# Eye Patch

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
[![Coverage][coverage-badge]][coverage-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/flint-crew/eye-patch/workflows/CI/badge.svg
[actions-link]:             https://github.com/flint-crew/eye-patch/actions
[pypi-link]:                https://pypi.org/project/eye-patch/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/eye-patch
[pypi-version]:             https://img.shields.io/pypi/v/eye-patch
[rtd-badge]:                https://readthedocs.org/projects/eye-patch/badge/?version=latest
[rtd-link]:                 https://eye-patch.readthedocs.io/en/latest/?badge=latest
[coverage-badge]:           https://codecov.io/github/flint-crew/eye-patch/branch/main/graph/badge.svg
[coverage-link]:            https://codecov.io/github/flint-crew/eye-patch

<!-- prettier-ignore-end -->

The pirate mask!

A python library and CLI tools for creating masks to assist deconvolution of interferometric data.

This library was originally the masking module of the [Flint](https://github.com/flint-crew/flint) pipeline, but has now been separated out for flexible use.

<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/2a8e442c-036b-4d78-8370-180c02588c95" />

## Documentation

Full documentation is provided on
[ReadtheDocs](https://eye-patch.readthedocs.io/).

## Installation

We publish releases on PyPI:

```bash
# PyPI release
pip install eye-patch
```

You can also install directly from the git repository:

```bash
# Direct git install (latest push)
pip install git+https://github.com/flint-crew/eye-patch.git
```

Or, from a local clone:

```bash
git clone https://github.com/flint-crew/eye-patch.git
cd flint
pip install -e .
```
We highly recommend using [uv](https://docs.astral.sh/uv/) for speedy installations.
Reproducible builds can be created using `uv sync`:

```bash
git clone https://github.com/flint-crew/eye-patch.git
uv venv
uv sync
```


## Contributions

Contributions are welcome! Please do submit a pull-request or issue if you spot
something you would like to address.

The full set of dev tooling can be installed via:

```bash
git clone https://github.com/flint-crew/eye-patch.git
cd eye-patch
pip install '.[dev]'
```

Or, using `uv`:

```bash
git clone https://github.com/flint-crew/eye-patch.git
cd flint
uv sync --group dev
```