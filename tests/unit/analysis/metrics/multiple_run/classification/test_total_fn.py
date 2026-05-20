# -*- coding: ascii -*-
"""
Tests for total fn.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.multiple_run.classification import TotalFN


class TestTotalFN:
    """Test cases for TotalFN."""

    def test_empty_runs_returns_zero(self, make_multiple_runs):
        """Empty runs should return 0."""
        metric = TotalFN(error_margin=(0, 0))
        runs = make_multiple_runs([])
        assert metric.evaluate(runs) == 0

    def test_single_run_no_fn(self, make_multiple_runs):
        """Run with no FN should return 0."""
        metric = TotalFN(error_margin=(0, 0))
        runs = make_multiple_runs([([10], [10])])
        assert metric.evaluate(runs) == 0

    def test_single_run_with_fn(self, make_multiple_runs):
        """Run with FN should count them."""
        metric = TotalFN(error_margin=(0, 0))
        runs = make_multiple_runs([([10], [10, 20])])
        assert metric.evaluate(runs) == 1

    def test_multiple_runs_sum(self, make_multiple_runs):
        """Should sum FN across runs."""
        metric = TotalFN(error_margin=(0, 0))
        runs = make_multiple_runs(
            [
                ([10], [10, 20]),
                ([30], [30, 40]),
            ]
        )
        assert metric.evaluate(runs) == 2

    def test_error_margin_affects_result(self, make_multiple_runs):
        """Error margin should affect matching - detection within margin should match."""
        metric = TotalFN(error_margin=(5, 5))
        runs = make_multiple_runs([([15], [10])])
        metric_no_margin = TotalFN(error_margin=(0, 0))
        runs2 = make_multiple_runs([([15], [10])])
        assert metric.evaluate(runs) == 0
        assert metric_no_margin.evaluate(runs2) == 1

    def test_result_is_int(self, make_multiple_runs):
        """Result should be int."""
        metric = TotalFN(error_margin=(0, 0))
        runs = make_multiple_runs([([10], [10])])
        assert isinstance(metric.evaluate(runs), int)

    def test_inherits_from_total_sum(self):
        """Should inherit from TotalSum."""
        from pysatl_cpd.analysis.metrics.multiple_run.aggregation_metric import TotalSum

        assert issubclass(TotalFN, TotalSum)
