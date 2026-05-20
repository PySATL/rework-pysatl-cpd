# -*- coding: ascii -*-
"""
Tests for delay metric.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.analysis.metrics.single_run.online.delays import Delays


class TestDelays:
    """Test cases for Delays metric."""

    def test_init_negative_max_delay_raises(self):
        """Negative max_delay should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            Delays(max_delay=-1)

    def test_perfect_detection(self, make_online_run):
        """Exact match should return delay of 0."""
        metric = Delays(max_delay=10)
        run = make_online_run([10], [10])
        result = metric.evaluate(run)
        assert result == [0]

    def test_delay_within_margin(self, make_online_run):
        """Detection after true CP within margin should return positive delay."""
        metric = Delays(max_delay=10)
        run = make_online_run([12], [10])
        result = metric.evaluate(run)
        assert result == [2]

    def test_missed_detection_returns_max_delay(self, make_online_run):
        """Missed true CP should return max_delay."""
        metric = Delays(max_delay=10)
        run = make_online_run([], [10])
        result = metric.evaluate(run)
        assert result == [10]

    def test_asymmetric_margin(self, make_online_run):
        """Only detections at or after true CP within max_delay match."""
        metric = Delays(max_delay=5)
        run = make_online_run([10], [12])
        result = metric.evaluate(run)
        assert result == [5]

    def test_multiple_true_cps(self, make_online_run):
        """Should return delay for each true CP."""
        metric = Delays(max_delay=10)
        run = make_online_run([12, 25], [10, 20])
        result = metric.evaluate(run)
        assert result == [2, 5]

    def test_multiple_detections_for_one_cp(self, make_online_run):
        """Should return minimum delay among matched detections."""
        metric = Delays(max_delay=10)
        run = make_online_run([12, 13, 14], [10])
        result = metric.evaluate(run)
        assert result == [2]

    def test_result_length_equals_true_cps_count(self, make_online_run):
        """Result list length should equal number of true CPs."""
        metric = Delays(max_delay=10)
        run = make_online_run([10, 20, 30], [5, 15, 25])
        result = metric.evaluate(run)
        assert len(result) == 3

    def test_result_is_list_of_int(self, make_online_run):
        """Result should be list of int."""
        metric = Delays(max_delay=10)
        run = make_online_run([10], [10])
        result = metric.evaluate(run)
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)
