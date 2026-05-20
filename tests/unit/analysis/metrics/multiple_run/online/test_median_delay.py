# -*- coding: ascii -*-
"""
Tests for median delay.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.analysis.metrics.multiple_run.online.delay import MedianDelayMetric


class TestMedianDelayMetric:
    """Test cases for MedianDelayMetric."""

    def test_init_negative_max_delay_raises(self):
        """Negative max_delay should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            MedianDelayMetric(max_delay=-1)

    def test_empty_runs_returns_max_delay(self, make_online_multiple_runs):
        """Empty runs should return max_delay."""
        metric = MedianDelayMetric(max_delay=10)
        runs = make_online_multiple_runs([])
        result = metric.evaluate(runs)
        assert result == 10.0

    def test_perfect_detection(self, make_online_multiple_runs):
        """Perfect detection: median delay = 0."""
        metric = MedianDelayMetric(max_delay=10)
        runs = make_online_multiple_runs([([10], [10])])
        result = metric.evaluate(runs)
        assert result == 0.0

    def test_single_delay(self, make_online_multiple_runs):
        """Single delay value."""
        metric = MedianDelayMetric(max_delay=10)
        runs = make_online_multiple_runs([([12], [10])])
        result = metric.evaluate(runs)
        assert result == 2.0

    def test_multiple_runs_median(self, make_online_multiple_runs):
        """Should compute median across all runs."""
        metric = MedianDelayMetric(max_delay=10)
        runs = make_online_multiple_runs(
            [
                ([12], [10]),
                ([14], [10]),
                ([16], [10]),
            ]
        )
        result = metric.evaluate(runs)
        assert result == 4.0

    def test_missed_detection_uses_max_delay(self, make_online_multiple_runs):
        """Missed CP contributes max_delay to median."""
        metric = MedianDelayMetric(max_delay=10)
        runs = make_online_multiple_runs([([], [10])])
        result = metric.evaluate(runs)
        assert result == 10.0

    def test_result_is_float(self, make_online_multiple_runs):
        """Result should be float."""
        metric = MedianDelayMetric(max_delay=10)
        runs = make_online_multiple_runs([([10], [10])])
        result = metric.evaluate(runs)
        assert isinstance(result, float)

    def test_inherits_from_total_median(self):
        """Should inherit from TotalMedian."""
        from pysatl_cpd.analysis.metrics.multiple_run.aggregation_metric import TotalMedian

        assert issubclass(MedianDelayMetric, TotalMedian)
