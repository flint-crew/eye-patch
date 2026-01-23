See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/


# Setting up a development environment manually

You can set up a development environment by running:

```bash
uv sync
```

# Pre-commit

You should prepare pre-commit or prek, which will help you by checking that
commits pass required checks:

```bash
uv tool install pre-commit # or brew install pre-commit on macOS
pre-commit install # Will install a pre-commit hook into the git repo
```

You can also/alternatively run `pre-commit run` (changes only) or
`pre-commit run --all-files` to check even without installing the hook.

# Testing

Use pytest to run the unit checks:

```bash
uv run pytest
```

# Coverage

Use pytest-cov to generate coverage reports:

```bash
uv run pytest --cov=eye-patch
```
