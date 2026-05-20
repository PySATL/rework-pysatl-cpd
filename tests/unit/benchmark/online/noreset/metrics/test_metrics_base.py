# -*- coding: ascii -*-
"""
Tests for metrics base.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping, Sequence

import pytest

from pysatl_cpd.analysis.metrics.abstracts.imultiple_run_metric import IMultipleRunMetric
from pysatl_cpd.analysis.metrics.abstracts.isingle_run_metric import ISingleRunMetric
from pysatl_cpd.analysis.metrics.multiple_run.derived_metric import DerivedMetric
from pysatl_cpd.benchmark.online.noreset.metrics.base import (
    NoResetDerivedMetric,
    NoResetMultipleRunMetric,
    NoResetSingleRunMetric,
    wrap_noreset_derived_metric,
    wrap_noreset_multiple_run_metric,
    wrap_noreset_single_run_metric,
)
from pysatl_cpd.benchmark.online.noreset.metrics.policy import NoResetPolicy

# ---------------------------------------------------------------------------
# NoResetSingleRunMetric
# ---------------------------------------------------------------------------


class _FakePolicy(NoResetPolicy):
    """Minimal policy that records calls and returns a fake transformed run."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, float]] = []

    def apply(self, run, threshold):
        self.calls.append((run, threshold))
        return "transformed_run"


class _FakeSingleMetric(ISingleRunMetric):
    """Minimal single-run metric that records its input."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    def evaluate(self, run):
        self.calls.append(run)
        return 42.0


def test_noreset_single_run_metric_init_stores_source_and_policy() -> None:
    source = _FakeSingleMetric()
    policy = _FakePolicy()
    metric = NoResetSingleRunMetric(source=source, policy=policy)
    assert metric.source is source
    assert metric.policy is policy


def test_noreset_single_run_metric_evaluate_returns_callable() -> None:
    source = _FakeSingleMetric()
    policy = _FakePolicy()
    metric = NoResetSingleRunMetric(source=source, policy=policy)
    run = object()
    evaluator = metric.evaluate(run)  # type: ignore[arg-type]
    assert callable(evaluator)


def test_noreset_single_run_metric_evaluator_calls_policy_and_source() -> None:
    source = _FakeSingleMetric()
    policy = _FakePolicy()
    metric = NoResetSingleRunMetric(source=source, policy=policy)
    run = object()
    evaluator = metric.evaluate(run)  # type: ignore[arg-type]

    result = evaluator(0.5)

    assert result == 42.0
    assert policy.calls == [(run, 0.5)]
    assert source.calls == ["transformed_run"]


def test_noreset_single_run_metric_evaluator_different_thresholds() -> None:
    source = _FakeSingleMetric()
    policy = _FakePolicy()
    metric = NoResetSingleRunMetric(source=source, policy=policy)
    run = object()
    evaluator = metric.evaluate(run)  # type: ignore[arg-type]

    evaluator(1.0)
    evaluator(2.0)

    assert policy.calls == [(run, 1.0), (run, 2.0)]


# ---------------------------------------------------------------------------
# NoResetMultipleRunMetric
# ---------------------------------------------------------------------------


class _FakeMultiMetric(IMultipleRunMetric):
    """Minimal multiple-run metric that records its input."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    def evaluate(self, runs):
        self.calls.append(list(runs))
        return [1.0, 2.0]


def test_noreset_multiple_run_metric_init_stores_source_and_policy() -> None:
    source = _FakeMultiMetric()
    policy = _FakePolicy()
    metric = NoResetMultipleRunMetric(source=source, policy=policy)
    assert metric.source is source
    assert metric.policy is policy


def test_noreset_multiple_run_metric_evaluate_returns_callable() -> None:
    source = _FakeMultiMetric()
    policy = _FakePolicy()
    metric = NoResetMultipleRunMetric(source=source, policy=policy)
    runs = [object(), object()]
    evaluator = metric.evaluate(runs)  # type: ignore[arg-type]
    assert callable(evaluator)


def test_noreset_multiple_run_metric_evaluator_transforms_all_runs() -> None:
    source = _FakeMultiMetric()
    policy = _FakePolicy()
    metric = NoResetMultipleRunMetric(source=source, policy=policy)
    run_a, run_b = object(), object()
    evaluator = metric.evaluate([run_a, run_b])  # type: ignore[arg-type]

    result = evaluator(0.5)

    assert result == [1.0, 2.0]
    assert policy.calls == [(run_a, 0.5), (run_b, 0.5)]
    assert source.calls == [["transformed_run", "transformed_run"]]


# ---------------------------------------------------------------------------
# NoResetDerivedMetric
# ---------------------------------------------------------------------------


class _FakeDerivedMetric(DerivedMetric):
    """Minimal derived metric with configurable bases."""

    def __init__(self, base_names: Sequence[str]) -> None:
        self._base_names = base_names
        self.compute_calls: list[object] = []

    @property
    def bases(self) -> Mapping[str, object]:
        return {name: object() for name in self._base_names}

    def compute(self, values):
        self.compute_calls.append(values)
        return sum(values.values())


class _FakeNoResetThresholdMetric:
    """Minimal no-reset threshold metric."""

    def __init__(self, evaluate_return: object = 10.0) -> None:
        self.evaluate_return = evaluate_return
        self.evaluate_calls: list[object] = []

    def evaluate(self, runs):
        self.evaluate_calls.append(runs)

        def evaluator(threshold: float) -> object:
            return self.evaluate_return

        return evaluator


def test_noreset_derived_metric_raises_on_missing_bases() -> None:
    source = _FakeDerivedMetric(base_names=["precision", "recall"])
    bases = {"precision": _FakeNoResetThresholdMetric()}
    with pytest.raises(ValueError, match="Missing no-reset bases for derived metric: recall"):
        NoResetDerivedMetric(source=source, bases=bases)  # type: ignore[arg-type]


def test_noreset_derived_metric_all_bases_present() -> None:
    source = _FakeDerivedMetric(base_names=["a", "b"])
    base_a = _FakeNoResetThresholdMetric(evaluate_return=5.0)
    base_b = _FakeNoResetThresholdMetric(evaluate_return=7.0)
    metric = NoResetDerivedMetric(source=source, bases={"a": base_a, "b": base_b})  # type: ignore[arg-type]
    assert metric.source is source
    assert metric.bases == {"a": base_a, "b": base_b}


def test_noreset_derived_metric_evaluate_combines_results() -> None:
    source = _FakeDerivedMetric(base_names=["a", "b"])
    base_a = _FakeNoResetThresholdMetric(evaluate_return=5.0)
    base_b = _FakeNoResetThresholdMetric(evaluate_return=7.0)
    metric = NoResetDerivedMetric(source=source, bases={"a": base_a, "b": base_b})  # type: ignore[arg-type]

    runs = [object(), object()]
    evaluator = metric.evaluate(runs)  # type: ignore[arg-type]

    result = evaluator(0.5)

    assert result == 12.0
    assert source.compute_calls == [{"a": 5.0, "b": 7.0}]


def test_noreset_derived_metric_evaluate_passes_threshold_to_bases() -> None:
    source = _FakeDerivedMetric(base_names=["x"])
    base_x = _FakeNoResetThresholdMetric(evaluate_return=99.0)
    metric = NoResetDerivedMetric(source=source, bases={"x": base_x})  # type: ignore[arg-type]

    evaluator = metric.evaluate([object()])  # type: ignore[arg-type]
    result = evaluator(1.5)

    assert result == 99.0


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_wrap_noreset_single_run_metric() -> None:
    source = _FakeSingleMetric()
    policy = _FakePolicy()
    wrapped = wrap_noreset_single_run_metric(source, policy)
    assert isinstance(wrapped, NoResetSingleRunMetric)
    assert wrapped.source is source
    assert wrapped.policy is policy


def test_wrap_noreset_multiple_run_metric() -> None:
    source = _FakeMultiMetric()
    policy = _FakePolicy()
    wrapped = wrap_noreset_multiple_run_metric(source, policy)
    assert isinstance(wrapped, NoResetMultipleRunMetric)
    assert wrapped.source is source
    assert wrapped.policy is policy


def test_wrap_noreset_derived_metric() -> None:
    source = _FakeDerivedMetric(base_names=["a"])
    bases = {"a": _FakeNoResetThresholdMetric()}
    wrapped = wrap_noreset_derived_metric(source, bases)  # type: ignore[arg-type]
    assert isinstance(wrapped, NoResetDerivedMetric)
    assert wrapped.source is source
    assert wrapped.bases is bases
