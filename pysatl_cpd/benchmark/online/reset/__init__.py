# -*- coding: ascii -*-
"""Reset-online benchmark public API.

Provides tools for benchmarking reset-online change-point detectors across
labeled datasets. A reset-online detector declares change points and then
resets its internal state, restarting the learning phase. This package
specialises the generic benchmark infrastructure for that semantics.

The public surface consists of three types:

- ``OnlineResetBenchmark`` -- Orchestrator that binds a dataset and a
  ``BenchmarkRegistry`` together and evaluates multiple-run metrics over
  a collection of detector entries. Inherits from the generic
  ``Benchmark`` class.
- ``OnlineResetBenchmarkEntry`` -- Dataclass that wraps an
  ``OnlineResetDetector`` and exposes its ``ChangePointDetectorDescription``
  for use in benchmark campaigns.
- ``OnlineResetWholeTimeseriesMetricScenario`` -- Scenario that runs each
  detector entry against every provider in a dataset, then evaluates a
  supplied ``IMultipleRunMetric`` over the collected runs.

Each submodule has its own docstring with implementation details:

- ``benchmark`` -- ``OnlineResetBenchmark`` orchestrator and its
  ``get_metrics_for`` convenience method.
- ``entry`` -- ``OnlineResetBenchmarkEntry`` dataclass.
- ``scenarios`` -- ``OnlineResetWholeTimeseriesMetricScenario`` for
  whole-timeseries metric evaluation.

Examples
--------
Create benchmark entries from a threshold sweep of reset detectors:

>>> from pysatl_cpd.algorithms.online import ShewhartControlChart
>>> from pysatl_cpd.benchmark.online.reset import OnlineResetBenchmarkEntry
>>> from pysatl_cpd.core.online import OnlineResetDetector
>>> from pysatl_cpd.data.providers.transformers import ColumnsSelectorTransformer
>>> feature_transformer = ColumnsSelectorTransformer(columns=["feature_0"])
>>> entries = [
...     OnlineResetBenchmarkEntry(
...         detector=OnlineResetDetector(
...             ShewhartControlChart(learning_period_size=100, window_size=10),
...             threshold=t,
...             skip_period=5,
...             data_transformer=feature_transformer,
...         ),
...     )
...     for t in [2.0, 3.0, 4.0]
... ]
>>> len(entries)
3

Run a benchmark against a preset dataset and evaluate a multiple-run metric:

>>> from pysatl_cpd.analysis.metrics.multiple_run.classification import ClassificationReport
>>> from pysatl_cpd.benchmark.online.reset import OnlineResetBenchmark
>>> from pysatl_cpd.benchmark.registry import BenchmarkRegistry
>>> from pysatl_cpd.data.generator import preset_dataset
>>> dataset = preset_dataset("extreme_mean_shifts", n_series=10, seed=7, series_length=500)
>>> registry = BenchmarkRegistry()
>>> benchmark = OnlineResetBenchmark(dataset=dataset, registry=registry)
>>> metric = ClassificationReport(error_margin=(0, 15))
>>> results = benchmark.get_metrics_for(entries, metric)
>>> len(results)
3

Notes
-----
- Detector runs are cached in the ``BenchmarkRegistry`` so that multiple
  metric evaluations over the same entries can reuse execution results
  without re-running detectors.
- Parallel execution is handled by joblib. The default backend is
  ``"loky"``. Pass ``n_jobs > 1`` to ``OnlineResetBenchmark`` or to
  ``get_metrics_for`` to enable parallelism.
- All metric objects must implement ``IMultipleRunMetric`` and accept
  ``OnlineDetectionTrace`` instances as input.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.benchmark.online.reset.benchmark import OnlineResetBenchmark
from pysatl_cpd.benchmark.online.reset.entry import OnlineResetBenchmarkEntry
from pysatl_cpd.benchmark.online.reset.scenarios import OnlineResetWholeTimeseriesMetricScenario

__all__ = [
    "OnlineResetBenchmark",
    "OnlineResetBenchmarkEntry",
    "OnlineResetWholeTimeseriesMetricScenario",
]
