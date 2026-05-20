# -*- coding: ascii -*-
"""
Tests for total fp.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.multiple_run.classification import TotalFP


class TestTotalFP:
    """Test cases for TotalFP."""

    def test_empty_runs_returns_zero(self, make_multiple_runs):
        """Empty runs should return 0."""
        metric = TotalFP(error_margin=(0, 0))
        runs = make_multiple_runs([])
        assert metric.evaluate(runs) == 0

    def test_single_run_no_fp(self, make_multiple_runs):
        """Run with no FP should return 0."""
        metric = TotalFP(error_margin=(0, 0))
        runs = make_multiple_runs([([10], [10])])
        assert metric.evaluate(runs) == 0

    def test_single_run_with_fp(self, make_multiple_runs):
        """Run with FP should count them."""
        metric = TotalFP(error_margin=(0, 0))
        runs = make_multiple_runs([([10, 20], [10])])
        assert metric.evaluate(runs) == 1

    def test_multiple_runs_sum(self, make_multiple_runs):
        """Should sum FP across runs."""
        metric = TotalFP(error_margin=(0, 0))
        runs = make_multiple_runs(
            [
                ([10, 20], [10]),
                ([30, 40], [30]),
            ]
        )
        assert metric.evaluate(runs) == 2

    def test_error_margin_affects_result(self, make_multiple_runs):
        """Error margin should affect matching."""
        metric = TotalFP(error_margin=(5, 5))
        runs = make_multiple_runs([([10, 20], [15])])
        assert metric.evaluate(runs) == 0

    def test_result_is_int(self, make_multiple_runs):
        """Result should be int."""
        metric = TotalFP(error_margin=(0, 0))
        runs = make_multiple_runs([([10], [10])])
        assert isinstance(metric.evaluate(runs), int)

    def test_inherits_from_total_sum(self):
        """Should inherit from TotalSum."""
        from pysatl_cpd.analysis.metrics.multiple_run.aggregation_metric import TotalSum

        assert issubclass(TotalFP, TotalSum)
