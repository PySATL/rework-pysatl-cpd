# -*- coding: ascii -*-
"""
Tests for single run metric contract.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.abstracts.isingle_run_metric import ISingleRunMetric


def test_single_run_metric_evaluate_is_abstract() -> None:
    assert getattr(ISingleRunMetric.evaluate, "__isabstractmethod__", False)


def test_single_run_metric_subclass_can_be_used_via_interface() -> None:
    class Metric(ISingleRunMetric):
        def evaluate(self, run) -> int:
            return 1

    assert Metric().evaluate(object()) == 1
