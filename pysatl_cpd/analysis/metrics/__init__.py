# -*- coding: ascii -*-
"""Evaluation metrics for change-point detection algorithms.

This package provides a comprehensive suite of metrics for evaluating both
offline and online change-point detection (CPD) algorithms. Metrics are
organized into two evaluation scopes: single-run metrics that assess one
detection trace against ground truth, and multiple-run metrics that
aggregate results across many runs for benchmarking.

.. raw:: html

    <h2>Public API</h2>

Abstract interfaces:

- ``ISingleRunMetric`` -- base class for metrics evaluated on one
  ``SingleRun``. See :mod:`pysatl_cpd.analysis.metrics.abstracts`.
- ``IMultipleRunMetric`` -- base class for metrics evaluated over a
  sequence of ``SingleRun`` objects. See
  :mod:`pysatl_cpd.analysis.metrics.abstracts`.

Single-run metrics (operate on one ``SingleRun``):

- ``ClassificationPrimitive`` -- base class for count-based classification
  metrics. See :mod:`pysatl_cpd.analysis.metrics.single_run`.
- ``TruePositiveCount`` -- counts true change points with matched detections.
- ``FalsePositiveCount`` -- counts unmatched detections.
- ``FalseNegativeCount`` -- counts true change points with no detection.
- ``Delays`` -- per-change-point detection delays for online algorithms.
- ``RunLengths`` -- distances between consecutive detections.

Multiple-run aggregation metrics (operate on a sequence of ``SingleRun``):

- ``AggregationMetric`` -- abstract base that reduces per-run results via a
  user-defined method. See :mod:`pysatl_cpd.analysis.metrics.multiple_run`.
- ``TotalSum`` -- sums per-run numeric results.
- ``TotalMean`` -- arithmetic mean of per-run numeric results.
- ``TotalMedian`` -- median of per-run numeric results.
- ``DerivedMetric`` -- combines multiple multi-run metric outputs.

Multiple-run classification metrics:

- ``TotalTP`` -- total true positives across all runs.
- ``TotalFP`` -- total false positives across all runs.
- ``TotalFN`` -- total false negatives across all runs.
- ``PrecisionMetric`` -- micro-averaged precision.
- ``RecallMetric`` -- micro-averaged recall.
- ``FScoreMetric`` -- F-beta score (F1 when ``beta=1``).
- ``ClassificationReport`` -- full classification summary dict.

Multiple-run online metrics:

- ``ARLMetric`` -- mean average run length across all runs.
- ``MeanDelayMetric`` -- mean detection delay across all runs.
- ``MedianDelayMetric`` -- median detection delay across all runs.

.. raw:: html

    <h2>Subpackages</h2>

- ``abstracts`` -- abstract base classes ``ISingleRunMetric`` and
  ``IMultipleRunMetric`` that define the evaluation protocol.
- ``single_run`` -- metrics for evaluating a single detection run, including
  classification counts and online timing metrics.
- ``multiple_run`` -- metrics that aggregate results over many runs,
  including classification summaries and online delay statistics.

Each subpackage has its own docstring with detailed examples and notes.

Notes
-----
- Classification metrics use an ``error_margin`` tuple ``(left, right)``
  to define a tolerance window around each true change point for matching
  detections.
- Multiple-run classification metrics use micro-averaging: counts are summed
  across all runs before ratios are computed.
- Online delay metrics require a ``max_delay`` parameter that caps both the
  matching window and the penalty for missed detections.
- All metrics are generic over trace type, provider type, and result type.
  Type parameters are inferred from the ``SingleRun`` passed to ``evaluate``.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.abstracts.imultiple_run_metric import IMultipleRunMetric
from pysatl_cpd.analysis.metrics.abstracts.isingle_run_metric import ISingleRunMetric
from pysatl_cpd.analysis.metrics.multiple_run.aggregation_metric import (
    AggregationMetric,
    TotalMean,
    TotalMedian,
    TotalSum,
)
from pysatl_cpd.analysis.metrics.multiple_run.classification import (
    ClassificationReport,
    FScoreMetric,
    PrecisionMetric,
    RecallMetric,
    TotalFN,
    TotalFP,
    TotalTP,
)
from pysatl_cpd.analysis.metrics.multiple_run.derived_metric import DerivedMetric
from pysatl_cpd.analysis.metrics.multiple_run.online import (
    ARLMetric,
    MeanDelayMetric,
    MedianDelayMetric,
)
from pysatl_cpd.analysis.metrics.single_run import (
    ClassificationPrimitive,
    Delays,
    FalseNegativeCount,
    FalsePositiveCount,
    RunLengths,
    TruePositiveCount,
)

__all__ = [
    "AggregationMetric",
    "ARLMetric",
    "ClassificationPrimitive",
    "ClassificationReport",
    "Delays",
    "DerivedMetric",
    "FalseNegativeCount",
    "FalsePositiveCount",
    "FScoreMetric",
    "IMultipleRunMetric",
    "ISingleRunMetric",
    "MeanDelayMetric",
    "MedianDelayMetric",
    "PrecisionMetric",
    "RecallMetric",
    "RunLengths",
    "TotalFN",
    "TotalFP",
    "TotalMean",
    "TotalMedian",
    "TotalSum",
    "TotalTP",
    "TruePositiveCount",
]
