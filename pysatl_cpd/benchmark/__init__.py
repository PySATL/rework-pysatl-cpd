# -*- coding: ascii -*-
"""Benchmark infrastructure for PySATL CPD.

Provides a layered framework for systematically evaluating change-point
detectors across labeled datasets. The package separates detector execution
from metric scoring through a registry-based caching model, enabling
multiple analyses to reuse the same detector runs without recomputation.

.. raw:: html

    <h2>Public API</h2>

- ``Benchmark`` -> Generic orchestrator that binds a ``Dataset`` and a
  ``BenchmarkRegistry`` together and executes scenarios. See the
  ``benchmark`` module docstring for details.
- ``BenchmarkRegistry`` -> Dict-like cache mapping ``SingleRunDescription``
  to ``SingleRun``. Supports parallel execution via joblib and pickle-based
  persistence. See the ``registry`` module docstring for details.
- ``BenchmarkScenario`` -> Abstract base class defining the scenario
  lifecycle: ``prepare_benchmark_jobs``, ``analyze``, and
  ``handle_benchmark_error``. See the ``scenarios`` module docstring.
- ``BenchmarkJob`` -> Frozen dataclass binding a ``ChangePointDetector`` to
  its assigned ``LabeledData`` providers. See the ``scenarios`` module
  docstring.
- ``DEFAULT_JOB_PARALLEL_BACKEND`` -> Default joblib backend string
  (``"loky"``). See the ``registry`` module docstring.
- ``online`` -> Subpackage for online detector benchmarking, split into
  ``noreset`` (continuous, policy-driven classification) and ``reset``
  (whole-timeseries event sequences). See the ``online`` subpackage
  docstring for the full API reference and usage examples.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Create a registry and populate it with detector runs::

    >>> from pysatl_cpd.algorithms.online import ShewhartControlChart
    >>> from pysatl_cpd.benchmark.registry import BenchmarkRegistry
    >>> from pysatl_cpd.core.online import OnlineResetDetector
    >>> from pysatl_cpd.data.generator import preset_dataset
    >>> from pysatl_cpd.data.providers.transformers import ColumnsSelectorTransformer
    >>>
    >>> dataset = preset_dataset("extreme_mean_shifts", n_series=4, seed=7, series_length=200)
    >>> transformer = ColumnsSelectorTransformer(columns=["feature_0"])
    >>> detector = OnlineResetDetector(
    ...     ShewhartControlChart(learning_period_size=50, window_size=10),
    ...     threshold=3.0,
    ...     skip_period=5,
    ...     data_transformer=transformer,
    ... )
    >>> registry = BenchmarkRegistry()
    >>> registry.update(detector, dataset[:2], n_jobs=1)
    >>> len(registry)
    2

Use the generic benchmark orchestrator with a custom scenario::

    >>> from pysatl_cpd.benchmark import Benchmark, BenchmarkJob, BenchmarkScenario
    >>> from pysatl_cpd.benchmark.registry import BenchmarkRegistry
    >>> from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
    >>> from pysatl_cpd.data.dataset import Dataset
    >>> from pysatl_cpd.data import LabeledData, TimeseriesAnnotation
    >>> from pysatl_cpd.core import DetectionTrace
    >>> from collections.abc import Sequence
    >>> from pysatl_cpd.data.generator import preset_dataset
    >>>
    >>> class CountingScenario(BenchmarkScenario):
    ...     def prepare_benchmark_jobs(self, dataset):
    ...         return [BenchmarkJob(detector=None, providers=list(dataset))]
    ...     def analyze(self, registry):
    ...         return {ChangePointDetectorDescription(name="dummy", parameters={}): len(registry)}
    >>>
    >>> dataset = preset_dataset("mean_shifts", n_series=2, seed=0, series_length=100)
    >>> benchmark = Benchmark(dataset=dataset, registry=BenchmarkRegistry())
    >>> # scenario requires a real detector; this demonstrates the orchestration pattern

Run an online reset benchmark via the ``online`` subpackage::

    >>> from pysatl_cpd.algorithms.online import ShewhartControlChart
    >>> from pysatl_cpd.benchmark.online.reset import (
    ...     OnlineResetBenchmark,
    ...     OnlineResetBenchmarkEntry,
    ... )
    >>> from pysatl_cpd.benchmark.registry import BenchmarkRegistry
    >>> from pysatl_cpd.core.online import OnlineResetDetector
    >>> from pysatl_cpd.data.generator import preset_dataset
    >>> from pysatl_cpd.data.providers.transformers import ColumnsSelectorTransformer
    >>>
    >>> dataset = preset_dataset("extreme_mean_shifts", n_series=4, seed=7, series_length=200)
    >>> transformer = ColumnsSelectorTransformer(columns=["feature_0"])
    >>> entries = [
    ...     OnlineResetBenchmarkEntry(
    ...         detector=OnlineResetDetector(
    ...             ShewhartControlChart(learning_period_size=50, window_size=10),
    ...             threshold=t,
    ...             skip_period=5,
    ...             data_transformer=transformer,
    ...         ),
    ...     )
    ...     for t in [2.0, 3.0, 4.0]
    ... ]
    >>> benchmark = OnlineResetBenchmark(dataset=dataset, registry=BenchmarkRegistry())
    >>> len(entries)
    3

.. raw:: html

    <h2>Notes</h2>

Notes
-----
- Detector runs are cached in the ``BenchmarkRegistry`` keyed by
  ``SingleRunDescription`` (detector description + provider description).
  Multiple metric evaluations reuse cached results without re-execution.
- Parallel execution uses joblib. The default backend is ``"loky"``.
  Pass ``n_jobs > 1`` to enable parallelism. Clone detectors via
  ``detector.clone()`` before use in parallel workers to ensure
  isolation.
- Registry persistence is opt-in. Call ``export_registry`` to serialize
  to a pickle file and ``upload_registry`` to load it back.
- The ``online`` subpackage provides specialised benchmark classes for
  both reset and no-reset online detectors. Reset detectors produce
  whole-timeseries event sequences; no-reset detectors run continuously
  with policy-driven classification at benchmark time.
"""

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.benchmark import online
from pysatl_cpd.benchmark.benchmark import Benchmark
from pysatl_cpd.benchmark.registry import DEFAULT_JOB_PARALLEL_BACKEND, BenchmarkRegistry
from pysatl_cpd.benchmark.scenarios import BenchmarkJob, BenchmarkScenario

__all__ = [
    "DEFAULT_JOB_PARALLEL_BACKEND",
    "Benchmark",
    "BenchmarkJob",
    "BenchmarkRegistry",
    "BenchmarkScenario",
    "online",
]
