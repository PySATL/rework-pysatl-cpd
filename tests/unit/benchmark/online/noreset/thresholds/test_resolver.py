# -*- coding: ascii -*-
"""
Tests for resolver.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry
from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import AutoThresholdsRange, ManualThresholdsRange
from pysatl_cpd.benchmark.online.noreset.thresholds.resolver import (
    NoResetThresholdResolver,
    ThresholdAutoTuneConfig,
    auto_pick_thresholds,
)
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.providers.labeled import LabeledData
from tests.support.algorithms import CountingAlgorithm, CountingAlgorithmConfig, CountingAlgorithmState
from tests.support.providers import make_univariate_labeled


def _make_run(values: list[float]) -> SingleRun[OnlineDetectionTrace[CountingAlgorithmState], LabeledData[Any, Any]]:
    trace = OnlineDetectionTrace(
        detector_description=ChangePointDetectorDescription(name="detector"),
        detected_change_points=[],
        threshold=None,
        processing_time=np.zeros(len(values), dtype=np.float64),
        detection_function=np.asarray(values, dtype=np.float64),
        algorithm_states=[CountingAlgorithmState() for _ in values],
    )
    return SingleRun(trace=trace, provider=make_univariate_labeled())


@dataclass
class _CallableMetric:
    fn: Any

    def evaluate(self, runs: Any) -> Any:
        del runs
        return self.fn


def test_auto_pick_thresholds_rejects_invalid_config() -> None:
    runs = [_make_run([0.0, 1.0])]

    def precision(threshold: float) -> float:
        return 0.0 if threshold < 0.5 else 1.0

    def recall(threshold: float) -> float:
        return 1.0 if threshold < 0.5 else 0.0

    with pytest.raises(ValueError, match="threshold_count"):
        auto_pick_thresholds(
            runs, lambda _: recall, lambda _: precision, config=ThresholdAutoTuneConfig(threshold_count=1)
        )
    with pytest.raises(ValueError, match="t_max"):
        auto_pick_thresholds(
            runs,
            lambda _: recall,
            lambda _: precision,
            config=ThresholdAutoTuneConfig(t_min=1.0, t_max=1.0),
        )
    with pytest.raises(ValueError, match="finite"):
        auto_pick_thresholds(
            runs,
            lambda _: recall,
            lambda _: precision,
            config=ThresholdAutoTuneConfig(t_min=0.0, t_max=float("nan")),
        )
    with pytest.raises(ValueError, match="bs_error"):
        auto_pick_thresholds(
            runs,
            lambda _: recall,
            lambda _: precision,
            config=ThresholdAutoTuneConfig(bs_error=0.0),
        )


def test_auto_pick_thresholds_computes_expected_bounds() -> None:
    result = auto_pick_thresholds(
        runs=[_make_run([0.0, 1.0])],
        make_recall=lambda _: lambda threshold: 1.0 if threshold < 0.6 else 0.0,
        make_precision=lambda _: lambda threshold: 0.0 if threshold < 0.3 else 1.0,
        config=ThresholdAutoTuneConfig(threshold_count=5, bs_error=1e-4, t_min=0.0, t_max=1.0),
    )

    assert result.min_threshold == pytest.approx(0.2999, rel=0.0, abs=5e-4)
    assert result.max_threshold == pytest.approx(0.6, rel=0.0, abs=5e-4)
    assert len(result.thresholds) == 5
    assert result.precision[0] == 0.0
    assert result.recall[-1] == 0.0


def test_resolve_classification_thresholds_uses_manual_range() -> None:
    entry = OnlineNoResetBenchmarkEntry(
        algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
        thresholds=ManualThresholdsRange(_values=(1.0, 2.5, 3.0)),
    )

    resolved = NoResetThresholdResolver().resolve_classification_thresholds(entry, [_make_run([0.0, 1.0])], report=None)  # type: ignore[arg-type]

    assert resolved == [1.0, 2.5, 3.0]


def test_resolve_classification_thresholds_auto_uses_report_metrics() -> None:
    entry = OnlineNoResetBenchmarkEntry(
        algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
        thresholds=AutoThresholdsRange(count=3),
    )
    report = type(
        "Report",
        (),
        {
            "bases": {
                "precision": _CallableMetric(lambda threshold: 0.0 if threshold < 0.25 else 1.0),
                "recall": _CallableMetric(lambda threshold: 1.0 if threshold < 0.75 else 0.0),
            }
        },
    )()

    resolved = NoResetThresholdResolver().resolve_classification_thresholds(entry, [_make_run([0.0, 2.0])], report)

    assert len(resolved) == 3
    assert resolved[0] < resolved[-1]
    assert resolved[0] == pytest.approx(0.25, abs=1e-3)
    assert resolved[-1] == pytest.approx(0.75, abs=1e-3)


def test_resolve_arl_thresholds_prefers_explicit_thresholds() -> None:
    entry = OnlineNoResetBenchmarkEntry(
        algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
        thresholds=AutoThresholdsRange(count=4),
    )

    resolved = NoResetThresholdResolver().resolve_arl_thresholds(entry, [_make_run([1.0, 2.0])], [2.0, 5.0])

    assert resolved == [2.0, 5.0]


def test_resolve_arl_thresholds_handles_empty_inputs_and_single_threshold() -> None:
    resolver = NoResetThresholdResolver()
    auto_entry = OnlineNoResetBenchmarkEntry(
        algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
        thresholds=AutoThresholdsRange(count=1),
    )
    manual_entry = OnlineNoResetBenchmarkEntry(
        algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
        thresholds=ManualThresholdsRange(_values=()),
    )

    with pytest.raises(ValueError, match="no runs provided"):
        resolver.resolve_arl_thresholds(auto_entry, [], None)
    with pytest.raises(ValueError, match="empty detection trace"):
        resolver.resolve_arl_thresholds(auto_entry, [_make_run([])], None)
    with pytest.raises(ValueError, match="no finite detection values"):
        resolver.resolve_arl_thresholds(auto_entry, [_make_run([float("nan"), float("inf")])], None)
    with pytest.raises(ValueError, match="empty thresholds_range"):
        resolver.resolve_arl_thresholds(manual_entry, [_make_run([1.0])], None)

    assert resolver.resolve_arl_thresholds(auto_entry, [_make_run([2.0, 6.0])], None) == [4.0]
    assert resolver.resolve_arl_thresholds(auto_entry, [_make_run([float("nan"), 2.0, 6.0])], None) == [4.0]


def test_infer_t_max_from_trace_values_uses_positive_fallback() -> None:
    resolver = NoResetThresholdResolver()

    assert resolver.infer_t_max_from_trace_values([_make_run([-2.0, -1.0])]) == 1e-12
    assert resolver.infer_t_max_from_trace_values([_make_run([float("nan"), 2.0])]) == pytest.approx(2.02)
    with pytest.raises(ValueError, match="no runs provided"):
        resolver.infer_t_max_from_trace_values([])
    with pytest.raises(ValueError, match="empty detection trace"):
        resolver.infer_t_max_from_trace_values([_make_run([])])
    with pytest.raises(ValueError, match="no finite detection values"):
        resolver.infer_t_max_from_trace_values([_make_run([float("nan"), float("inf")])])


def test_resolve_classification_thresholds_auto_handles_nan_detection_values() -> None:
    entry = OnlineNoResetBenchmarkEntry(
        algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
        thresholds=AutoThresholdsRange(count=3),
    )
    report = type(
        "Report",
        (),
        {
            "bases": {
                "precision": _CallableMetric(lambda threshold: 0.0 if threshold < 0.25 else 1.0),
                "recall": _CallableMetric(lambda threshold: 1.0 if threshold < 0.75 else 0.0),
            }
        },
    )()

    resolved = NoResetThresholdResolver().resolve_classification_thresholds(
        entry, [_make_run([float("nan"), 2.0])], report
    )

    assert len(resolved) == 3
    assert not any(np.isnan(resolved))
