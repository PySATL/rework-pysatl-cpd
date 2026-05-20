# -*- coding: ascii -*-
"""
Tests for mean delay.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.analysis.metrics.multiple_run.online.delay import MeanDelayMetric


class TestMeanDelayMetric:
    """Test cases for MeanDelayMetric."""

    def test_init_negative_max_delay_raises(self):
        """Negative max_delay should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            MeanDelayMetric(max_delay=-1)

    def test_empty_runs_returns_max_delay(self, make_online_multiple_runs):
        """Empty runs should return max_delay."""
        metric = MeanDelayMetric(max_delay=10)
        runs = make_online_multiple_runs([])
        result = metric.evaluate(runs)
        assert result == 10.0

    def test_perfect_detection(self, make_online_multiple_runs):
        """Perfect detection: mean delay = 0."""
        metric = MeanDelayMetric(max_delay=10)
        runs = make_online_multiple_runs([([10], [10])])
        result = metric.evaluate(runs)
        assert result == 0.0

    def test_delayed_detection(self, make_online_multiple_runs):
        """Delayed detection: mean of delays."""
        metric = MeanDelayMetric(max_delay=10)
        runs = make_online_multiple_runs([([12], [10])])
        result = metric.evaluate(runs)
        assert result == 2.0

    def test_multiple_runs_mean(self, make_online_multiple_runs):
        """Should compute mean across all runs."""
        metric = MeanDelayMetric(max_delay=10)
        runs = make_online_multiple_runs(
            [
                ([12], [10]),
                ([14], [10]),
            ]
        )
        result = metric.evaluate(runs)
        assert result == 3.0

    def test_missed_detection_uses_max_delay(self, make_online_multiple_runs):
        """Missed CP contributes max_delay to mean."""
        metric = MeanDelayMetric(max_delay=10)
        runs = make_online_multiple_runs([([], [10])])
        result = metric.evaluate(runs)
        assert result == 10.0

    def test_result_is_float(self, make_online_multiple_runs):
        """Result should be float."""
        metric = MeanDelayMetric(max_delay=10)
        runs = make_online_multiple_runs([([10], [10])])
        result = metric.evaluate(runs)
        assert isinstance(result, float)

    def test_inherits_from_total_mean(self):
        """Should inherit from TotalMean."""
        from pysatl_cpd.analysis.metrics.multiple_run.aggregation_metric import TotalMean

        assert issubclass(MeanDelayMetric, TotalMean)
