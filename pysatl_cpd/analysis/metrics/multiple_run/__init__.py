# -*- coding: ascii -*-
"""Base classes for aggregating single-run metrics across multiple runs.

This module provides the foundational building blocks for computing metrics
over collections of change-point detection runs. It defines two abstract
patterns -- aggregation and derivation -- along with ready-to-use numeric
aggregators.

.. raw:: html

    <h2>Public API</h2>

- ``AggregationMetric`` -- abstract base that evaluates a single-run metric
  on each run and reduces the per-run results via a user-defined ``aggregate``
  method. See :mod:`aggregation_metric` for details.
- ``TotalSum`` -- sums flat or nested per-run numeric results into one total.
- ``TotalMean`` -- computes the global arithmetic mean of per-run numeric
  results.
- ``TotalMedian`` -- computes the global median of per-run numeric results.
- ``DerivedMetric`` -- abstract base that evaluates several multi-run metrics
  on the same runs and combines their outputs via a user-defined ``compute``
  method. See :mod:`derived_metric` for details.

Examples
--------
Subclass ``TotalMean`` to wrap a single-run metric:

>>> from typing import Any
>>> from pysatl_cpd.analysis.metrics.multiple_run import TotalMean
>>> from pysatl_cpd.analysis.metrics.single_run import TruePositiveCount
>>> from pysatl_cpd.core.detection_trace import DetectionTrace
>>> from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData
>>> class MeanTP[TraceT: DetectionTrace, ProviderT: LabeledData[Any, Any]](
...     TotalMean[TraceT, ProviderT, int]
... ):
...     def __init__(self, error_margin: tuple[int, int]) -> None:
...         self._base = TruePositiveCount[TraceT, ProviderT](error_margin)
...     @property
...     def base_metric(self) -> TruePositiveCount[TraceT, ProviderT]:
...         return self._base
>>> metric = MeanTP(error_margin=(0, 15))

Evaluate the aggregated metric over a sequence of runs:

>>> # runs = [SingleRun(trace1, provider1), SingleRun(trace2, provider2), ...]
>>> # result = metric.evaluate(runs)

Subclass ``DerivedMetric`` to combine multiple multi-run metrics:

>>> from collections.abc import Mapping
>>> from pysatl_cpd.analysis.metrics.multiple_run import DerivedMetric
>>> from pysatl_cpd.analysis.metrics.abstracts import IMultipleRunMetric
>>> from pysatl_cpd.typedefs import Number
>>> class CustomDerived[TraceT: DetectionTrace, ProviderT: LabeledData[Any, Any]](
...     DerivedMetric[TraceT, ProviderT, Number, float]
... ):
...     def __init__(self, error_margin: tuple[int, int]) -> None:
...         from pysatl_cpd.analysis.metrics.multiple_run.classification import (
...             TotalTP,
...             TotalFP,
...         )
...         self._bases: Mapping[str, IMultipleRunMetric[TraceT, ProviderT, int]] = {
...             "tp": TotalTP[TraceT, ProviderT](error_margin),
...             "fp": TotalFP[TraceT, ProviderT](error_margin),
...         }
...     @property
...     def bases(self) -> Mapping[str, IMultipleRunMetric[TraceT, ProviderT, int]]:
...         return self._bases
...     def compute(self, values: Mapping[str, Number]) -> float:
...         tp = values["tp"]
...         fp = values["fp"]
...         return float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0

Notes
-----
- ``TotalMean`` and ``TotalMedian`` raise ``ValueError`` on empty input
  sequences. Subclasses can override ``_value_on_empty`` to return a
  fallback value instead.
- All aggregation classes handle both flat and nested (per-change-point)
  numeric results by flattening them before reduction.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.multiple_run.aggregation_metric import (
    AggregationMetric,
    TotalMean,
    TotalMedian,
    TotalSum,
)
from pysatl_cpd.analysis.metrics.multiple_run.derived_metric import DerivedMetric

__all__ = [
    "AggregationMetric",
    "DerivedMetric",
    "TotalMean",
    "TotalMedian",
    "TotalSum",
]
