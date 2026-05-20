# -*- coding: ascii -*-
"""Multiple-run online metrics.

Aggregated evaluation metrics for online change-point detection algorithms
across collections of reset-benchmark runs.

This module provides three metric classes that summarize detector behaviour
over multiple runs: average run length (ARL), mean detection delay, and
median detection delay. Each class is an aggregation metric -- it delegates
per-run scoring to a single-run metric and then reduces the per-run results
to a single scalar. For the aggregation protocol and base classes, see the
``pysatl_cpd.analysis.metrics.multiple_run`` subpackage. For the underlying
single-run metrics, see ``pysatl_cpd.analysis.metrics.single_run.online``.

.. raw:: html

    <h2>Public API</h2>

- ``ARLMetric`` -- Computes the mean distance between consecutive detected
  change points across all runs. Ground truth is not used; every detection
  is treated as a positive. Based on ``RunLengths``.
- ``MeanDelayMetric`` -- Computes the mean detection delay across all runs.
  Delay is the time between a true change point and its matched detection.
  Missed change points are penalised with a configurable ``max_delay`` cap.
  Based on ``Delays``.
- ``MedianDelayMetric`` -- Same as ``MeanDelayMetric`` but reports the
  median rather than the mean. Based on ``Delays``.

.. raw:: html

    <h2>Examples</h2>

Delay metrics require a ``max_delay`` parameter that caps both the matching
window and the penalty for missed detections. ARL has no parameters.

Evaluate metrics directly on a sequence of runs::

    >>> from pysatl_cpd.analysis.metrics.multiple_run.online import (
    ...     ARLMetric,
    ...     MeanDelayMetric,
    ...     MedianDelayMetric,
    ... )
    >>> from pysatl_cpd.core.single_run import SingleRun
    >>> # runs: Sequence[SingleRun[OnlineDetectionTrace, LabeledData]]
    >>> arl = ARLMetric().evaluate(runs)  # doctest: +SKIP
    >>> mean_delay = MeanDelayMetric(max_delay=100).evaluate(runs)  # doctest: +SKIP
    >>> median_delay = MedianDelayMetric(max_delay=100).evaluate(runs)  # doctest: +SKIP

Use metrics through the reset benchmark API::

    >>> from pysatl_cpd.analysis.metrics.multiple_run.online import ARLMetric
    >>> from pysatl_cpd.benchmark.online.reset import OnlineResetBenchmark
    >>> benchmark = OnlineResetBenchmark(registry=registry)  # doctest: +SKIP
    >>> results = benchmark.get_metrics_for(entries, ARLMetric())  # doctest: +SKIP

.. raw:: html

    <h2>Notes</h2>

- ``ARLMetric`` ignores ground-truth labels. It measures alarm frequency,
  not detection accuracy. Use it alongside classification metrics for a
  complete picture.
- ``MeanDelayMetric`` and ``MedianDelayMetric`` raise ``ValueError`` if
  ``max_delay`` is negative.
- All three classes inherit from ``AggregationMetric`` and follow the
  ``IMultipleRunMetric.evaluate`` protocol. See the base class docstrings
  in ``pysatl_cpd.analysis.metrics.multiple_run.aggregation_metric`` for
  details on how per-run results are collected and reduced.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.multiple_run.online.arl import ARLMetric
from pysatl_cpd.analysis.metrics.multiple_run.online.delay import MeanDelayMetric, MedianDelayMetric

__all__ = [
    "ARLMetric",
    "MeanDelayMetric",
    "MedianDelayMetric",
]
