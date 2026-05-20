# -*- coding: ascii -*-
"""
Tests for multiple run metric contract.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.abstracts.imultiple_run_metric import IMultipleRunMetric


def test_multiple_run_metric_evaluate_is_abstract() -> None:
    assert getattr(IMultipleRunMetric.evaluate, "__isabstractmethod__", False)


def test_multiple_run_metric_subclass_can_be_used_via_interface() -> None:
    class Metric(IMultipleRunMetric):
        def evaluate(self, runs) -> int:
            return len(runs)

    assert Metric().evaluate([1, 2, 3]) == 3
