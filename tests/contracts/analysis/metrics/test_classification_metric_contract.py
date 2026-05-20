# -*- coding: ascii -*-
"""
Tests for classification metric contract.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.single_run.classification.base import ClassificationMetricBase, ClassificationPrimitive


class _MatchedCountMetric(ClassificationPrimitive):
    def evaluate(self, run) -> int:
        matched = self.match(run.trace.detected_change_points, run.provider.change_points)
        return sum(bool(indices) for indices in matched.values())


def test_classification_metric_base_evaluate_is_abstract() -> None:
    assert getattr(ClassificationMetricBase.evaluate, "__isabstractmethod__", False)


def test_classification_primitive_evaluate_is_abstract() -> None:
    assert getattr(ClassificationPrimitive.evaluate, "__isabstractmethod__", False)


def test_classification_metric_base_match_uses_error_margin(make_classification_run) -> None:
    run = make_classification_run(detected=[8], true_cps=[10])
    metric = _MatchedCountMetric(error_margin=(2, 0))
    assert metric.evaluate(run) == 1


def test_classification_metric_base_match_rejects_detection_outside_margin(make_classification_run) -> None:
    run = make_classification_run(detected=[7], true_cps=[10])
    metric = _MatchedCountMetric(error_margin=(0, 2))
    assert metric.evaluate(run) == 0
