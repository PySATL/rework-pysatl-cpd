# -*- coding: ascii -*-
"""Online benchmark subpackage.

Provides infrastructure for benchmarking online change-point detectors
across labeled datasets. The package is split into two subpackages that
handle the two fundamental online detector semantics: no-reset detectors
that run continuously without restarting after an alarm, and reset
detectors that restart their internal state after each declared change
point.

.. raw:: html

    <h2>Public API</h2>

- ``noreset`` -> No-reset online benchmarking. Exposes
  ``OnlineNoResetBenchmark``, ``OnlineNoResetBenchmarkEntry``, detector
  types, classification policies, metrics, scenarios, threshold ranges,
  and tooling. See the ``noreset`` subpackage docstring for the full API
  reference and usage examples.
- ``reset`` -> Reset-online benchmarking. Exposes ``OnlineResetBenchmark``,
  ``OnlineResetBenchmarkEntry``, and ``OnlineResetWholeTimeseriesMetricScenario``.
  See the ``reset`` subpackage docstring for details and examples.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
No-reset benchmark with a threshold sweep::

    >>> from pysatl_cpd.algorithms.online import ShewhartControlChart
    >>> from pysatl_cpd.benchmark.online import noreset
    >>> from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import (
    ...     LinearThresholdsRange,
    ... )
    >>> from pysatl_cpd.benchmark.online.noreset.tooling.bisegment_cut import (
    ...     BisegmentCut,
    ... )
    >>> from pysatl_cpd.benchmark.registry import BenchmarkRegistry
    >>> from pysatl_cpd.data.generator import preset_dataset
    >>> from pysatl_cpd.data.providers.transformers import ColumnsSelectorTransformer
    >>>
    >>> dataset = preset_dataset("mean_shifts", n_series=4, seed=42, series_length=120)
    >>> transformer = ColumnsSelectorTransformer(columns=["feature_0"])
    >>> entry = noreset.OnlineNoResetBenchmarkEntry(
    ...     algorithm=ShewhartControlChart(learning_period_size=20, window_size=10),
    ...     thresholds=LinearThresholdsRange(start=1.5, end=3.0, count=4),
    ...     data_transformer=transformer,
    ...     bisegment_cut=BisegmentCut.parse((8, 0)),
    ... )
    >>> benchmark = noreset.OnlineNoResetBenchmark(
    ...     dataset=dataset,
    ...     registry=BenchmarkRegistry(),
    ...     max_delay=15,
    ...     global_policy=noreset.NoResetPolicyKind.MIXED,
    ...     error_margin=(0, 15),
    ...     policy_strict=False,
    ... )
    >>> tables = benchmark.get_classification_table([entry])
    >>> len(tables)
    1

Reset benchmark with multiple detector entries::

    >>> from pysatl_cpd.algorithms.online import ShewhartControlChart
    >>> from pysatl_cpd.benchmark.online import reset
    >>> from pysatl_cpd.benchmark.registry import BenchmarkRegistry
    >>> from pysatl_cpd.core.online import OnlineResetDetector
    >>> from pysatl_cpd.data.generator import preset_dataset
    >>> from pysatl_cpd.data.providers.transformers import ColumnsSelectorTransformer
    >>>
    >>> dataset = preset_dataset("extreme_mean_shifts", n_series=10, seed=7, series_length=500)
    >>> transformer = ColumnsSelectorTransformer(columns=["feature_0"])
    >>> entries = [
    ...     reset.OnlineResetBenchmarkEntry(
    ...         detector=OnlineResetDetector(
    ...             ShewhartControlChart(learning_period_size=100, window_size=10),
    ...             threshold=t,
    ...             skip_period=5,
    ...             data_transformer=transformer,
    ...         ),
    ...     )
    ...     for t in [2.0, 3.0, 4.0]
    ... ]
    >>> benchmark = reset.OnlineResetBenchmark(
    ...     dataset=dataset,
    ...     registry=BenchmarkRegistry(),
    ... )
    >>> len(entries)
    3

.. raw:: html

    <h2>Notes</h2>

Notes
-----
- The ``noreset`` subpackage treats thresholds as evaluation settings
  applied to continuous traces at benchmark time. Entries bundle one
  algorithm with a ``ThresholdsRange`` rather than one fully thresholded
  detector.
- The ``reset`` subpackage wraps fully configured ``OnlineResetDetector``
  instances. Each entry corresponds to one concrete detector with a fixed
  threshold.
- Both subpackages share the ``BenchmarkRegistry`` for caching detector
  runs so that multiple metric evaluations reuse execution results.
- Parallel execution uses joblib with the ``"loky"`` backend by default.
  Pass ``n_jobs > 1`` to enable parallelism.
- Clone detectors via ``detector.clone()`` before use in parallel workers
  to ensure each worker holds an isolated algorithm instance.
"""

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.benchmark.online import noreset, reset

__all__ = ["noreset", "reset"]
