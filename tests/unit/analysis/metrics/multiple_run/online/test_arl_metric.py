# -*- coding: ascii -*-
"""
Tests for arl metric.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.analysis.metrics.multiple_run.online.arl import ARLMetric


class TestARLMetric:
    """Test cases for ARLMetric."""

    def test_empty_runs_raises(self, make_online_multiple_runs):
        """Empty runs should raise ValueError."""
        metric = ARLMetric()
        runs = make_online_multiple_runs([])
        with pytest.raises(ValueError, match="empty sequence"):
            metric.evaluate(runs)

    def test_single_run_single_detection(self, make_online_multiple_runs):
        """Single run with one detection."""
        metric = ARLMetric()
        runs = make_online_multiple_runs([([10], [5])])
        result = metric.evaluate(runs)
        assert isinstance(result, float)
        assert result == 10.0

    def test_single_run_multiple_detections(self, make_online_multiple_runs):
        """RunLengths returns multiple values, ARL averages them."""
        metric = ARLMetric()
        runs = make_online_multiple_runs([([3, 7, 10], [5, 15])])
        result = metric.evaluate(runs)
        # RunLengths: diff([0, 3, 7, 10]) = [3, 4, 3], mean = 3.333...
        assert abs(result - 3.333) < 0.01

    def test_multiple_runs_average(self, make_online_multiple_runs):
        """Should average run lengths across all runs."""
        metric = ARLMetric()
        runs = make_online_multiple_runs(
            [
                ([5], [10]),
                ([10], [10]),
            ]
        )
        result = metric.evaluate(runs)
        assert result == 7.5

    def test_no_detections(self, make_online_multiple_runs):
        """No detections: RunLengths returns [len(provider)]."""
        metric = ARLMetric()
        runs = make_online_multiple_runs([([], [10])], length=100)
        result = metric.evaluate(runs)
        assert result == 100.0

    def test_result_is_float(self, make_online_multiple_runs):
        """Result should be float."""
        metric = ARLMetric()
        runs = make_online_multiple_runs([([10], [5])])
        result = metric.evaluate(runs)
        assert isinstance(result, float)

    def test_inherits_from_total_mean(self):
        """Should inherit from TotalMean."""
        from pysatl_cpd.analysis.metrics.multiple_run.aggregation_metric import TotalMean

        assert issubclass(ARLMetric, TotalMean)
