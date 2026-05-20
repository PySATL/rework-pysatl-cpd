# AGENTS.md

## Source Of Truth
- Prefer `pyproject.toml`, `.pre-commit-config.yaml`, and `.github/workflows/ci.yaml` over `README.md`. The README still contains stale import paths and outdated examples.
- Current public package entrypoints are `pysatl_cpd.data`, `pysatl_cpd.data.generator`, and `pysatl_cpd.core.online`.

## Environment And Tooling
- Use Python 3.12+ locally. Repo config targets 3.12 (`ruff`, `mypy`, pre-commit), while CI runs on 3.12 and 3.13.
- Install with dev tools: `poetry install --with dev --no-interaction`
- CI order is:
  1. `poetry install --with dev --no-interaction`
  2. `poetry run pre-commit run --all-files`
  3. `poetry run pytest --cov`

## High-Value Verification Commands
- Full test suite: `poetry run pytest`
- Single test file: `poetry run pytest tests/unit/analysis/test_labeled_data.py`
- Pre-commit locally: `poetry run pre-commit run --all-files`
- Type check: `poetry run mypy`
- Ruff check: `poetry run ruff check .`
- Ruff format: `poetry run ruff format .`

## Repo-Specific Conventions
- Change-point indices are zero-based everywhere. Do not introduce 1-based indexing in providers, traces, benchmarks, tests, or notebooks.
- The old top-level generator package is gone. Import generators from `pysatl_cpd.data.generator`, not `pysatl_cpd.generator`.
- `pysatl_cpd.data` exports labeled-data providers, datasets, loaders, and transformers.
- `pysatl_cpd.data.generator` exports synthetic generators and should be imported directly.
- `pysatl_cpd.core.online` is the main online-CPD API surface (`OnlineCpdSolver`, `OnlineDetectionTrace`, algorithm interfaces).

## Structure That Matters
- `pysatl_cpd/data/`
  - main data-layer package
  - labeled providers, datasets, loaders, transformers, Huawei loaders
- `pysatl_cpd/data/generator/`
  - synthetic segment/labeled-data/dataset generators
- `pysatl_cpd/core/online/`
  - online solver, online trace, algorithm interfaces
- `tests/`
  - primary executable spec; trust tests over README prose

## Examples And Scripts
- Visualization examples: `./scripts/run_visualization_examples.sh`
  - wrapper sets `MPLBACKEND=Agg`
- Algorithm examples: `./scripts/run_algorithm_examples.sh`
  - wrapper sets `MPLBACKEND=Agg`
- Bisegment runner: `poetry run python scripts/algorithm_bisegment_runner/shewhart.py`

## Pre-commit Quirks
- Pre-commit excludes `*.md` and `*.ipynb`, so documentation and notebook changes are not validated there automatically.
- Pre-commit runs `ruff --fix`, `ruff-format`, `mypy`, and basic file checks.

## Testing / Coverage Notes
- `pytest` always runs with coverage because `addopts` is set in `pyproject.toml`.
- Coverage omits `pysatl_cpd/analysis/visualization/*`, so missing coverage there is expected.
- For data-layer changes, start with focused tests in `tests/unit/analysis`, `tests/unit/core`, and related benchmark tests before running the whole suite.

## Workflow Notes
- Commit style in `CONTRIBUTING.md`: conventional commits.
- Branches should not be committed directly to `main`.
- Project preference in PR flow is rebase, not merge.
