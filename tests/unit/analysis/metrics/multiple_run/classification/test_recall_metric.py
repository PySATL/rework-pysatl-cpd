# -*- coding: ascii -*-
"""
Tests for recall metric.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.multiple_run.classification.recall import RecallMetric


class TestRecallMetric:
    """Test cases for RecallMetric."""

    def test_empty_runs_returns_zero(self, make_multiple_runs):
        """Empty runs should return 0.0 (no detections)."""
        metric = RecallMetric(error_margin=(0, 0))
        runs = make_multiple_runs([])
        assert metric.evaluate(runs) == 0.0

    def test_perfect_recall(self, make_multiple_runs):
        """All true CPs detected: recall = 1.0."""
        metric = RecallMetric(error_margin=(0, 0))
        runs = make_multiple_runs([([10, 20], [10, 20])])
        assert metric.evaluate(runs) == 1.0

    def test_zero_recall(self, make_multiple_runs):
        """No true CPs detected: recall = 0.0."""
        metric = RecallMetric(error_margin=(0, 0))
        runs = make_multiple_runs([([10, 20], [30, 40])])
        assert metric.evaluate(runs) == 0.0

    def test_partial_recall(self, make_multiple_runs):
        """Half true CPs detected: recall = 0.5."""
        metric = RecallMetric(error_margin=(0, 0))
        runs = make_multiple_runs([([10], [10, 20])])
        assert metric.evaluate(runs) == 0.5

    def test_multiple_runs_micro_average(self, make_multiple_runs):
        """Should compute micro-averaged recall across runs."""
        metric = RecallMetric(error_margin=(0, 0))
        runs = make_multiple_runs(
            [
                ([10], [10, 20]),
                ([30], [30, 40]),
            ]
        )
        result = metric.evaluate(runs)
        assert result == 0.5

    def test_error_margin_affects_result(self, make_multiple_runs):
        """Error margin should affect matching - detection within margin should match."""
        metric = RecallMetric(error_margin=(5, 5))
        runs = make_multiple_runs([([15], [10])])
        assert metric.evaluate(runs) == 1.0

    def test_result_is_float(self, make_multiple_runs):
        """Result should be float."""
        metric = RecallMetric(error_margin=(0, 0))
        runs = make_multiple_runs([([10], [10])])
        assert isinstance(metric.evaluate(runs), float)

    def test_inherits_from_derived_metric(self):
        """Should inherit from DerivedMetric."""
        from pysatl_cpd.analysis.metrics.multiple_run.derived_metric import DerivedMetric

        assert issubclass(RecallMetric, DerivedMetric)
