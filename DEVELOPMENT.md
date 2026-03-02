# Development Guide

This guide is for contributors and developers who want to work on Heretic itself.

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-capable GPU (recommended for testing)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/p-e-w/heretic.git
   cd heretic
   ```

2. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Install dependencies:
   ```bash
   # Install all dependencies including optional research extras and dev tools
   uv sync --all-extras --dev
   ```

4. Run Heretic in development mode:
   ```bash
   uv run heretic --help
   ```

## Code Quality

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

### Linting

Check for code issues:
```bash
uv run ruff check src/
```

Auto-fix issues where possible:
```bash
uv run ruff check --fix src/
```

### Formatting

Check formatting:
```bash
uv run ruff format --check src/
```

Auto-format code:
```bash
uv run ruff format src/
```

### Import Sorting

Ruff also handles import sorting. Check import order:
```bash
uv run ruff check --select I src/
```

## Building

Build the package:
```bash
uv build
```

This creates distribution files in the `dist/` directory.

## Project Structure

```
heretic/
├── src/heretic/           # Main package
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── evaluator.py       # Model evaluation
│   ├── main.py            # Entry point
│   ├── model.py           # Model loading and abliteration
│   ├── utils.py           # Utility functions
├── config.default.toml    # Default configuration
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies
├── Dockerfile             # Docker image
├── docker-compose.yml     # Docker Compose config
└── environment.yml        # Conda environment
```

## Continuous Integration

The CI pipeline (`.github/workflows/ci.yml`) runs on every push and pull request:

1. **Formatting check**: Ensures code follows Ruff formatting rules
2. **Linting**: Checks for code issues and import sorting
3. **Build**: Verifies the package builds correctly

Make sure all checks pass before submitting a pull request.

## Adding Dependencies

### Core Dependencies

Add to the `dependencies` list in `pyproject.toml`:
```toml
dependencies = [
    "new-package>=1.0.0",
]
```

### Optional Dependencies

Add to `[project.optional-dependencies]`:
```toml
[project.optional-dependencies]
feature_name = [
    "optional-package>=1.0.0",
]
```

### Development Dependencies

Add to `[dependency-groups]`:
```toml
[dependency-groups]
dev = [
    "dev-tool>=1.0.0",
]
```

After adding dependencies, run:
```bash
uv sync --all-extras --dev
```

## Testing

Currently, Heretic does not have a formal test suite. Testing is primarily done manually:

1. Test basic functionality:
   ```bash
   uv run heretic --help
   ```

2. Test with a small model (if you have GPU access):
   ```bash
   uv run heretic --model your-model-name
   ```

## Release Process

1. Update version in `pyproject.toml`
2. Ensure all CI checks pass
3. Build the package: `uv build`
4. Tag the release: `git tag v1.x.x`
5. Push tags: `git push --tags`

## License

All contributions are released under the AGPL-3.0-or-later license.
