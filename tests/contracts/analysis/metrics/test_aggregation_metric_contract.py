# -*- coding: ascii -*-
"""
Tests for aggregation metric contract.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.abstracts.isingle_run_metric import ISingleRunMetric
from pysatl_cpd.analysis.metrics.multiple_run.aggregation_metric import (
    AggregationMetric,
    TotalMean,
    TotalMedian,
    TotalSum,
)


class _RunLengthMetric(ISingleRunMetric):
    def evaluate(self, run) -> int:
        return len(run.trace.detected_change_points)


class _NestedRunMetric(ISingleRunMetric):
    def evaluate(self, run) -> list[int]:
        return [len(run.trace.detected_change_points), len(run.provider.change_points)]


class _TotalRunLength(TotalSum):
    @property
    def base_metric(self) -> ISingleRunMetric:
        return _RunLengthMetric()


class _TotalNestedRunLength(TotalSum):
    @property
    def base_metric(self) -> ISingleRunMetric:
        return _NestedRunMetric()


class _MeanRunLength(TotalMean):
    @property
    def base_metric(self) -> ISingleRunMetric:
        return _RunLengthMetric()


class _MedianRunLength(TotalMedian):
    @property
    def base_metric(self) -> ISingleRunMetric:
        return _RunLengthMetric()


def test_aggregation_metric_requires_base_metric_and_aggregate() -> None:
    assert getattr(AggregationMetric.base_metric.fget, "__isabstractmethod__", False)
    assert getattr(AggregationMetric.aggregate, "__isabstractmethod__", False)


def test_total_sum_evaluate_aggregates_base_metric_across_runs(make_multiple_runs) -> None:
    runs = make_multiple_runs([([1, 2], [10]), ([5], [20])])
    assert _TotalRunLength().evaluate(runs) == 3


def test_total_sum_flattens_nested_results(make_multiple_runs) -> None:
    runs = make_multiple_runs([([1, 2], [10]), ([5], [20, 30])])
    assert _TotalNestedRunLength().evaluate(runs) == 6


def test_total_mean_raises_on_empty_input_without_fallback() -> None:
    try:
        _MeanRunLength().aggregate([])
    except ValueError as exc:
        assert "empty sequence" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty mean aggregation")


def test_total_mean_uses_fallback_when_configured() -> None:
    class Metric(_MeanRunLength):
        @property
        def _value_on_empty(self) -> float:
            return 7.0

    assert Metric().aggregate([]) == 7.0


def test_total_median_raises_on_empty_input_without_fallback() -> None:
    try:
        _MedianRunLength().aggregate([])
    except ValueError as exc:
        assert "empty sequence" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty median aggregation")


def test_total_median_uses_fallback_when_configured() -> None:
    class Metric(_MedianRunLength):
        @property
        def _value_on_empty(self) -> float:
            return 9.0

    assert Metric().aggregate([]) == 9.0
