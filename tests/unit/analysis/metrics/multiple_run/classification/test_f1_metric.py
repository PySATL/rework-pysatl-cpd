"""
Tests for f1 metric.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.multiple_run.classification.fmeasure import FScoreMetric


class TestF1Metric:
    """Test cases for FScoreMetric."""

    def test_empty_runs_returns_zero(self, make_multiple_runs):
        """Empty runs should return 0.0."""
        metric = FScoreMetric(error_margin=(0, 0))
        runs = make_multiple_runs([])
        assert metric.evaluate(runs) == 0.0

    def test_perfect_f1(self, make_multiple_runs):
        """Perfect precision and recall: F1 = 1.0."""
        metric = FScoreMetric(error_margin=(0, 0))
        runs = make_multiple_runs([([10, 20], [10, 20])])
        assert metric.evaluate(runs) == 1.0

    def test_zero_f1(self, make_multiple_runs):
        """Zero precision or recall: F1 = 0.0."""
        metric = FScoreMetric(error_margin=(0, 0))
        runs = make_multiple_runs([([10], [20])])
        assert metric.evaluate(runs) == 0.0

    def test_precision_1_recall_05(self, make_multiple_runs):
        """Precision=1.0, Recall=0.5: F1 = 2/3."""
        metric = FScoreMetric(error_margin=(0, 0))
        runs = make_multiple_runs([([10], [10, 20])])
        result = metric.evaluate(runs)
        assert abs(result - 2 / 3) < 1e-10

    def test_multiple_runs(self, make_multiple_runs):
        """Should compute F1 over all runs (micro-averaged)."""
        metric = FScoreMetric(error_margin=(0, 0))
        runs = make_multiple_runs(
            [
                ([10], [10, 20]),
                ([30, 40], [30, 50]),
            ]
        )
        result = metric.evaluate(runs)
        # tp=2, fp=1, fn=2
        # precision = 2/3 ~= 0.6667, recall = 2/4 = 0.5
        # f1 = 2 * 0.6667 * 0.5 / (0.6667 + 0.5) = 0.5714
        assert abs(result - 0.5714) < 0.01

    def test_non_default_beta(self, make_multiple_runs):
        """Non-default beta should weight recall differently."""
        metric = FScoreMetric(error_margin=(0, 0), beta=2.0)
        runs = make_multiple_runs([([10], [10, 20])])
        result = metric.evaluate(runs)
        assert abs(result - 5 / 9) < 1e-10

    def test_error_margin_affects_result(self, make_multiple_runs):
        """Error margin should affect matching."""
        metric = FScoreMetric(error_margin=(5, 5))
        runs = make_multiple_runs([([10], [15])])
        assert metric.evaluate(runs) == 1.0

    def test_negative_beta_raises(self):
        """Negative beta should be rejected."""
        try:
            FScoreMetric(error_margin=(0, 0), beta=-1.0)
        except ValueError as exc:
            assert str(exc) == "beta must be non-negative"
        else:
            raise AssertionError("Expected ValueError for negative beta")

    def test_result_is_float(self, make_multiple_runs):
        """Result should be float."""
        metric = FScoreMetric(error_margin=(0, 0))
        runs = make_multiple_runs([([10], [10])])
        assert isinstance(metric.evaluate(runs), float)

    def test_inherits_from_derived_metric(self):
        """Should inherit from DerivedMetric."""
        from pysatl_cpd.analysis.metrics.multiple_run.derived_metric import DerivedMetric

        assert issubclass(FScoreMetric, DerivedMetric)
