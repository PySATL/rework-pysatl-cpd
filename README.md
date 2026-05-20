# PySATL-CPD

[![mypy](https://img.shields.io/github/actions/workflow/status/PySATL/rework-pysatl-cpd/.github/workflows/ci.yaml?label=mypy&style=flat-square)](https://github.com/PySATL/rework-pysatl-cpd/actions/workflows/ci.yaml)
[![Tests](https://img.shields.io/github/actions/workflow/status/PySATL/rework-pysatl-cpd/.github/workflows/ci.yaml?label=Tests&style=flat-square)](https://github.com/PySATL/rework-pysatl-cpd/actions/workflows/ci.yaml)
[![Coverage](https://img.shields.io/coverallsCoverage/github/PySATL/rework-pysatl-cpd?style=flat-square)](https://coveralls.io/github/PySATL/rework-pysatl-cpd)
[![ruff](https://img.shields.io/github/actions/workflow/status/PySATL/rework-pysatl-cpd/.github/workflows/ci.yaml?label=ruff&style=flat-square)](https://github.com/PySATL/rework-pysatl-cpd/actions/workflows/ci.yaml)
[![pydoclint](https://img.shields.io/github/actions/workflow/status/PySATL/rework-pysatl-cpd/.github/workflows/ci.yaml?label=pydoclint&style=flat-square)](https://github.com/PySATL/rework-pysatl-cpd/actions/workflows/ci.yaml)
[![Docs](https://img.shields.io/github/actions/workflow/status/PySATL/rework-pysatl-cpd/.github/workflows/docs-deploy.yaml?label=Docs&style=flat-square)](https://github.com/PySATL/rework-pysatl-cpd/actions/workflows/docs-deploy.yaml)
[![MIT License](https://img.shields.io/github/license/PySATL/rework-pysatl-cpd?style=flat-square&color=blue)](LICENSE)

> **Note:** This repository was migrated from a private repository. The commit history has been preserved but does not reflect the actual development timeline.

PySATL **Change Point Detection** subproject (*abbreviated pysatl-cpd*) is a Python library for detecting change points in time series data — significant deviations from expected patterns or trends. Change points mark moments when the underlying statistical properties of a process shift, making them crucial for monitoring and analysis in finance, healthcare, network security, and many other domains.

This is a public mirror of the [original pysatl-cpd](https://github.com/PySATL/pysatl-cpd) repository.

## Algorithms

### Online algorithms

- **Bayesian Online CPD** — non-parametric Bayesian approach with configurable likelihood, hazard, and change-point function (MaxRunLength / Drop)
- **Shewhart Control Chart** — classical SPC method based on standardized deviation of a sliding-window mean
- **CUSUM family** — cumulative sum detectors:
  - Page Two-Sided CUSUM
  - Crosier CUSUM
  - Autoregressive CUSUM
  - Variance Two-Sided CUSUM

### Offline algorithms

> Offline algorithms are not yet available, but will be added in a future release.

### Data generation

- Synthetic scenario-based generators with configurable segment distributions (Normal, Student-t, Exponential, Uniform, Multivariate Normal)
- Preset scenarios: `mean_shifts`, `variance_shifts`, `covariance_shifts`, `extreme_mean_shifts`, `3d_mean_shifts`, `mixed_shifts`, `no_shifts`

---

## Requirements

- Python 3.13+
- Poetry 2.1.0+

## Installation

Clone the repository and install dependencies:

```sh
git clone https://github.com/PySATL/pysatl-cpd.git
cd pysatl-cpd
poetry install
```

For development with linting, type-checking, and testing tools:

```sh
poetry install --with dev
```

---

## Quick Start

Generate synthetic data, run a Shewhart detector, and visualize the result:

```python
import matplotlib.pyplot as plt
import numpy as np

from pysatl_cpd.algorithms.online import ShewhartControlChart
from pysatl_cpd.analysis.visualization import DrawBackend, OnlineCpdPlotter
from pysatl_cpd.core.online import OnlineResetDetector
from pysatl_cpd.data.generator import preset_dataset
from pysatl_cpd.data.providers.transformers import ColumnsSelectorTransformer

# Generate a preset dataset with mean shifts
dataset = preset_dataset("mean_shifts", n_series=1, seed=42, series_length=600)
data = dataset[0]

# Select a single feature for the univariate detector
feature_transformer = ColumnsSelectorTransformer(columns=["feature_0"])

# Configure and run Shewhart reset detector
algorithm = ShewhartControlChart(learning_period_size=30, window_size=10)
detector = OnlineResetDetector(
    algorithm, threshold=2.0, skip_period=5, collect_states=True,
    data_transformer=feature_transformer,
)
trace = detector.detect(data)

# Visualize
plotter = OnlineCpdPlotter(
    backend=DrawBackend.MATPLOTLIB,
    data_provider=data,
    detection_trace=trace,
)

# Draw true change points
plotter.set_ground_truth(list(data.change_points), margin=10)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig = plotter.draw(
    figure=fig,
    axes={"timeseries": axes[0], "detection_function": axes[1], "processing_time": axes[2]},
)
plt.tight_layout()
plt.show()
```

![Shewhart detection with OnlineCpdPlotter](assets/shewhart_detection.png)



---

## Project Structure

| Path | Description |
|------|-------------|
| `pysatl_cpd/data/` | Data layer: providers, datasets, loaders, transformers |
| `pysatl_cpd/data/generator/` | Synthetic generators and scenario specs |
| `pysatl_cpd/core/online/` | Online CPD API: `OnlineResetDetector`, `OnlineDetectionTrace`, algorithm interfaces |
| `pysatl_cpd/algorithms/online/` | Algorithm implementations: Bayesian, Shewhart, CUSUM |
| `pysatl_cpd/benchmark/` | Benchmarking: reset (`OnlineResetBenchmark`) and no-reset (`OnlineNoResetBenchmark`) |
| `pysatl_cpd/analysis/` | Metrics and visualization utilities |
| `tests/` | Test suite (the authoritative spec) |

---

## Development

### Install dev dependencies

```bash
poetry install --with dev
```

### Pre-commit hooks

```shell
poetry run pre-commit install
```

Run manually on all files:

```shell
poetry run pre-commit run --all-files --color always --verbose --show-diff-on-failure
```

### Run tests

```bash
poetry run pytest
```

Single test file:

```bash
poetry run pytest tests/unit/analysis/test_labeled_data.py
```

### Type check and lint

```bash
poetry run mypy
poetry run ruff check .
poetry run ruff format .
```

### Build documentation

```bash
cd docs
make html
```

Open `docs/build/html/index.html` in a browser to view the generated docs.

---

## License

This project is licensed under the terms of the **MIT** license. See the [LICENSE](LICENSE) for more information.
