# -*- coding: ascii -*-
"""
Tests for classification report.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.multiple_run.classification.report import ClassificationReport


class TestClassificationReport:
    """Test cases for ClassificationReport."""

    def test_empty_runs(self, make_multiple_runs):
        """Empty runs should return all zeros and default precision/recall/f1."""
        metric = ClassificationReport(error_margin=(0, 0))
        runs = make_multiple_runs([])
        result = metric.evaluate(runs)
        assert result["tp"] == 0
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["precision"] == 1.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_perfect_detection(self, make_multiple_runs):
        """All true CPs detected with no FP."""
        metric = ClassificationReport(error_margin=(0, 0))
        runs = make_multiple_runs([([10, 20], [10, 20])])
        result = metric.evaluate(runs)
        assert result["tp"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_partial_detection(self, make_multiple_runs):
        """Some true CPs missed, no FP (all detections match)."""
        metric = ClassificationReport(error_margin=(0, 0))
        runs = make_multiple_runs([([10, 30], [10, 20, 30])])
        result = metric.evaluate(runs)
        assert result["tp"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 1
        assert result["precision"] == 1.0
        assert abs(result["recall"] - 2 / 3) < 1e-10
        assert abs(result["f1"] - 0.8) < 1e-10

    def test_multiple_runs(self, make_multiple_runs):
        """Should aggregate across runs."""
        metric = ClassificationReport(error_margin=(0, 0))
        runs = make_multiple_runs(
            [
                ([10], [10, 20]),
                ([30, 40], [30, 50]),
            ]
        )
        result = metric.evaluate(runs)
        assert result["tp"] == 2
        assert result["fp"] == 1
        assert result["fn"] == 2

    def test_result_keys(self, make_multiple_runs):
        """Result should have all expected keys."""
        metric = ClassificationReport(error_margin=(0, 0))
        runs = make_multiple_runs([([10], [10])])
        result = metric.evaluate(runs)
        expected_keys = {"tp", "fp", "fn", "precision", "recall", "f1"}
        assert set(result.keys()) == expected_keys

    def test_error_margin_affects_result(self, make_multiple_runs):
        """Error margin should affect matching."""
        metric = ClassificationReport(error_margin=(5, 5))
        runs = make_multiple_runs([([10], [15])])
        result = metric.evaluate(runs)
        assert result["tp"] == 1
        assert result["fp"] == 0
        assert result["fn"] == 0

    def test_inherits_from_derived_metric(self):
        """Should inherit from DerivedMetric."""
        from pysatl_cpd.analysis.metrics.multiple_run.derived_metric import DerivedMetric

        assert issubclass(ClassificationReport, DerivedMetric)
