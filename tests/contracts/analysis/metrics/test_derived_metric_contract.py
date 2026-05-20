# -*- coding: ascii -*-
"""
Tests for derived metric contract.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.abstracts.imultiple_run_metric import IMultipleRunMetric
from pysatl_cpd.analysis.metrics.multiple_run.derived_metric import DerivedMetric


class _ConstantMetric(IMultipleRunMetric):
    def __init__(self, value: int) -> None:
        self._value = value

    def evaluate(self, runs) -> int:
        return self._value + len(runs)


class _CombinedMetric(DerivedMetric):
    @property
    def bases(self):
        return {"left": _ConstantMetric(1), "right": _ConstantMetric(2)}

    def compute(self, values):
        return values["left"] + values["right"]


def test_derived_metric_requires_bases_and_compute() -> None:
    assert getattr(DerivedMetric.bases.fget, "__isabstractmethod__", False)
    assert getattr(DerivedMetric.compute, "__isabstractmethod__", False)


def test_derived_metric_evaluate_delegates_to_bases_and_compute(make_multiple_runs) -> None:
    runs = make_multiple_runs([([1], [10]), ([2], [20])])
    assert _CombinedMetric().evaluate(runs) == 7
