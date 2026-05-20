# -*- coding: ascii -*-
"""
Tests for tp metric.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.single_run.classification.tp_metric import TruePositiveCount


class TestTruePositiveCount:
    """Test cases for TruePositiveCount."""

    def test_no_detections_no_true_cps(self, make_classification_run):
        """No detections and no true CPs should return 0."""
        metric = TruePositiveCount(error_margin=(0, 0))
        run = make_classification_run([], [])
        assert metric.evaluate(run) == 0

    def test_detections_no_true_cps(self, make_classification_run):
        """Detections with no true CPs should return 0 TP."""
        metric = TruePositiveCount(error_margin=(0, 0))
        run = make_classification_run([10, 20, 30], [])
        assert metric.evaluate(run) == 0

    def test_no_detections_with_true_cps(self, make_classification_run):
        """No detections with true CPs should return 0 TP."""
        metric = TruePositiveCount(error_margin=(0, 0))
        run = make_classification_run([], [10, 20, 30])
        assert metric.evaluate(run) == 0

    def test_exact_matches(self, make_classification_run):
        """Exact matches should count each true CP as TP."""
        metric = TruePositiveCount(error_margin=(0, 0))
        run = make_classification_run([10, 20, 30], [10, 20, 30])
        assert metric.evaluate(run) == 3

    def test_detections_within_margin(self, make_classification_run):
        """Detections within error margin should count as TP."""
        metric = TruePositiveCount(error_margin=(5, 5))
        run = make_classification_run([10, 25], [5, 20])
        assert metric.evaluate(run) == 2

    def test_multiple_detections_per_true_cp(self, make_classification_run):
        """Multiple detections for one true CP: only counts as 1 TP."""
        metric = TruePositiveCount(error_margin=(5, 5))
        run = make_classification_run([10, 11, 12], [10])
        assert metric.evaluate(run) == 1

    def test_not_all_true_cps_matched(self, make_classification_run):
        """Only matched true CPs count as TP."""
        metric = TruePositiveCount(error_margin=(0, 0))
        run = make_classification_run([10], [10, 20, 30])
        assert metric.evaluate(run) == 1

    def test_error_margin_negative_raises(self):
        """Negative error margin should raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="non-negative"):
            TruePositiveCount(error_margin=(0, -1))

    def test_result_is_int(self, make_classification_run):
        """Result should be int."""
        metric = TruePositiveCount(error_margin=(0, 0))
        run = make_classification_run([10], [10])
        assert isinstance(metric.evaluate(run), int)
