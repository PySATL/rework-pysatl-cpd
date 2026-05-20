# -*- coding: ascii -*-
"""
Tests for fp metric.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.single_run.classification.fp_metric import FalsePositiveCount


class TestFalsePositiveCount:
    """Test cases for FalsePositiveCount."""

    def test_no_detections_no_true_cps(self, make_classification_run):
        """No detections and no true CPs should return 0."""
        metric = FalsePositiveCount(error_margin=(0, 0))
        run = make_classification_run([], [])
        assert metric.evaluate(run) == 0

    def test_detections_no_true_cps(self, make_classification_run):
        """All detections are FP when there are no true CPs."""
        metric = FalsePositiveCount(error_margin=(0, 0))
        run = make_classification_run([10, 20, 30], [])
        assert metric.evaluate(run) == 3

    def test_no_detections_with_true_cps(self, make_classification_run):
        """No detections with true CPs should return 0 FP."""
        metric = FalsePositiveCount(error_margin=(0, 0))
        run = make_classification_run([], [10, 20, 30])
        assert metric.evaluate(run) == 0

    def test_exact_matches(self, make_classification_run):
        """Exact matches should produce 0 FP."""
        metric = FalsePositiveCount(error_margin=(0, 0))
        run = make_classification_run([10, 20, 30], [10, 20, 30])
        assert metric.evaluate(run) == 0

    def test_detections_outside_margin(self, make_classification_run):
        """Detections outside error margin are FP."""
        metric = FalsePositiveCount(error_margin=(2, 2))
        run = make_classification_run([10, 25], [5, 20])
        assert metric.evaluate(run) == 2

    def test_detections_within_margin(self, make_classification_run):
        """Detections within error margin should not be FP."""
        metric = FalsePositiveCount(error_margin=(5, 5))
        run = make_classification_run([10, 25], [5, 20])
        assert metric.evaluate(run) == 0

    def test_multiple_detections_per_true_cp(self, make_classification_run):
        """Multiple detections for one true CP: all match within margin, so FP=0."""
        metric = FalsePositiveCount(error_margin=(5, 5))
        run = make_classification_run([10, 11, 12], [10])
        assert metric.evaluate(run) == 0

    def test_error_margin_negative_raises(self):
        """Negative error margin should raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="non-negative"):
            FalsePositiveCount(error_margin=(0, -1))

    def test_result_is_int(self, make_classification_run):
        """Result should be int."""
        metric = FalsePositiveCount(error_margin=(0, 0))
        run = make_classification_run([10], [10])
        assert isinstance(metric.evaluate(run), int)
