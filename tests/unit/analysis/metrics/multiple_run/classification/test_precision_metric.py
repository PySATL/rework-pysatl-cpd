# -*- coding: ascii -*-
"""
Tests for precision metric.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.multiple_run.classification.precision import PrecisionMetric


class TestPrecisionMetric:
    """Test cases for PrecisionMetric."""

    def test_empty_runs_returns_one(self, make_multiple_runs):
        """Empty runs should return 1.0 (no detections)."""
        metric = PrecisionMetric(error_margin=(0, 0))
        runs = make_multiple_runs([])
        assert metric.evaluate(runs) == 1.0

    def test_perfect_precision(self, make_multiple_runs):
        """All detections match true CPs: precision = 1.0."""
        metric = PrecisionMetric(error_margin=(0, 0))
        runs = make_multiple_runs([([10, 20], [10, 20])])
        assert metric.evaluate(runs) == 1.0

    def test_zero_precision(self, make_multiple_runs):
        """No detections match: precision = 0.0."""
        metric = PrecisionMetric(error_margin=(0, 0))
        runs = make_multiple_runs([([10, 20], [30, 40])])
        assert metric.evaluate(runs) == 0.0

    def test_partial_precision(self, make_multiple_runs):
        """Half detections match: precision = 0.5."""
        metric = PrecisionMetric(error_margin=(0, 0))
        runs = make_multiple_runs([([10, 20], [10, 30])])
        assert metric.evaluate(runs) == 0.5

    def test_multiple_runs_micro_average(self, make_multiple_runs):
        """Should compute micro-averaged precision across runs."""
        metric = PrecisionMetric(error_margin=(0, 0))
        runs = make_multiple_runs(
            [
                ([10, 20], [10, 30]),
                ([30, 40], [40, 50]),
            ]
        )
        result = metric.evaluate(runs)
        assert result == 0.5

    def test_error_margin_affects_result(self, make_multiple_runs):
        """Error margin should affect matching."""
        metric = PrecisionMetric(error_margin=(5, 5))
        runs = make_multiple_runs([([10], [15])])
        assert metric.evaluate(runs) == 1.0

    def test_result_is_float(self, make_multiple_runs):
        """Result should be float."""
        metric = PrecisionMetric(error_margin=(0, 0))
        runs = make_multiple_runs([([10], [10])])
        assert isinstance(metric.evaluate(runs), float)

    def test_inherits_from_derived_metric(self):
        """Should inherit from DerivedMetric."""
        from pysatl_cpd.analysis.metrics.multiple_run.derived_metric import DerivedMetric

        assert issubclass(PrecisionMetric, DerivedMetric)
