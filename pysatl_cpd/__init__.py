# -*- coding: ascii -*-
"""PySATL CPD -- Python library for Change Point Detection.

A comprehensive framework for detecting, evaluating, and benchmarking change
points in univariate and multivariate time series. The package covers the
full workflow: synthetic data generation, offline and online detection
algorithms, quantitative evaluation metrics, visualization, and systematic
benchmarking with registry-based caching.

The top-level package re-exports the complete public API from six subpackages.
Each subpackage has its own detailed docstring with examples; this module
provides a high-level map and cross-package usage patterns.

.. raw:: html

    <h2>Subpackages</h2>

- ``data`` -- Data-layer abstractions: providers (unlabeled and labeled),
  datasets, CSV loaders, type definitions, and a synthetic data generator.
  See the :mod:`~pysatl_cpd.data` docstring for the full API.
- ``core`` -- Fundamental building blocks: abstract detector interfaces,
  unified result containers, single-run analysis helpers, and the full
  online detection API (``core.online``). See the :mod:`~pysatl_cpd.core`
  docstring for details.
- ``algorithms`` -- Online algorithm implementations: Bayesian BOCPD,
  CUSUM family (Page, Crosier, variance, autoregressive), and Shewhart
  control charts. All implement the ``OnlineAlgorithm`` interface. See
  the :mod:`~pysatl_cpd.algorithms` docstring for details.
- ``analysis`` -- Evaluation metrics (single-run and multi-run classification,
  delays, run lengths) and visualization (time series plotters, trace
  visualizers, benchmark plotters). Supports Matplotlib and Plotly backends.
  See the :mod:`~pysatl_cpd.analysis` docstring for details.
- ``benchmark`` -- Systematic benchmarking infrastructure: registry-based
  caching, scenario orchestration, and online benchmark subpackages for
  both reset and no-reset detector semantics. See the
  :mod:`~pysatl_cpd.benchmark` docstring for details.
- ``typedefs`` -- Low-level shared types: numeric and array type aliases,
  an immutable ``frozendict``, and stable hashing utilities. See the
  :mod:`~pysatl_cpd.typedefs` docstring for details.

.. raw:: html

    <h2>Examples</h2>

Examples
--------

Generate a synthetic dataset and run an online detector::

    >>> import numpy as np
    >>> from pysatl_cpd.data.generator import preset_dataset
    >>> from pysatl_cpd.algorithms import ShewhartControlChart
    >>> from pysatl_cpd.core import OnlineResetDetector
    >>> from pysatl_cpd.data.providers.transformers import ColumnsSelectorTransformer
    >>> dataset = preset_dataset(
    ...     "mean_shifts", n_series=2, seed=42, series_length=120
    ... )
    >>> transformer = ColumnsSelectorTransformer(columns=["feature_0"])
    >>> provider = dataset[0]
    >>> detector = OnlineResetDetector(
    ...     ShewhartControlChart(learning_period_size=20, window_size=10),
    ...     threshold=2.5,
    ...     data_transformer=transformer,
    ... )
    >>> trace = detector.detect(provider)
    >>> len(trace.detected_change_points) >= 0
    True

Evaluate detection quality with single-run metrics::

    >>> from pysatl_cpd.core import SingleRun
    >>> from pysatl_cpd.analysis import TruePositiveCount, FalsePositiveCount
    >>> run = SingleRun(trace=trace, provider=provider)
    >>> tp = TruePositiveCount(error_margin=(0, 10)).evaluate(run)
    >>> fp = FalsePositiveCount(error_margin=(0, 10)).evaluate(run)
    >>> tp >= 0
    True
    >>> fp >= 0
    True

Benchmark multiple detectors with a shared registry::

    >>> from pysatl_cpd.benchmark import BenchmarkRegistry
    >>> registry = BenchmarkRegistry()
    >>> detectors = [
    ...     OnlineResetDetector(
    ...         ShewhartControlChart(learning_period_size=20, window_size=10),
    ...         threshold=t,
    ...         data_transformer=transformer,
    ...     )
    ...     for t in [2.0, 3.0, 4.0]
    ... ]
    >>> for det in detectors:
    ...     registry.update(det, dataset[:1], n_jobs=1)
    >>> len(registry)
    3

Use immutable frozendict and stable hashing::

    >>> from pysatl_cpd.typedefs import frozendict, stable_hash
    >>> cfg = frozendict(alpha=1.0, beta=2.0)
    >>> cfg["alpha"]
    1.0
    >>> stable_hash(cfg) >= 0
    True

.. raw:: html

    <h2>Notes</h2>

Notes
-----
- Python 3.12+ is required. The codebase uses PEP 695 generic syntax.
- All change-point indices are zero-based throughout the project.
- Install with development dependencies via
  ``poetry install --with dev --no-interaction``.
- Visualization requires ``matplotlib`` and ``plotly``. The ``arch``
  package is optional and needed only for ``AutoregressiveCUSUM``.
- Pre-commit runs ``ruff --fix``, ``ruff-format``, ``mypy``, and basic
  file checks. Documentation and notebook files are excluded from
  pre-commit.
- Tests are the authoritative spec; run with ``poetry run pytest``.
"""

__author__ = "Mikhail Mikhailov, Andrey Isakov, Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd import algorithms, analysis, benchmark, core, data, typedefs
from pysatl_cpd.algorithms import *
from pysatl_cpd.algorithms import __all__ as _algorithms_all
from pysatl_cpd.analysis import *
from pysatl_cpd.analysis import __all__ as _analysis_all
from pysatl_cpd.benchmark import *
from pysatl_cpd.benchmark import __all__ as _benchmark_all
from pysatl_cpd.core import *  # type: ignore[no-redef]
from pysatl_cpd.core import __all__ as _core_all
from pysatl_cpd.data import *
from pysatl_cpd.data import __all__ as _data_all
from pysatl_cpd.typedefs import *
from pysatl_cpd.typedefs import __all__ as _typedefs_all

__all__ = [
    "algorithms",
    "analysis",
    "benchmark",
    "core",
    "data",
    "typedefs",
    *_algorithms_all,
    *_analysis_all,
    *_benchmark_all,
    *_core_all,
    *_data_all,
    *_typedefs_all,
]

del _algorithms_all
del _analysis_all
del _benchmark_all
del _core_all
del _data_all
del _typedefs_all
