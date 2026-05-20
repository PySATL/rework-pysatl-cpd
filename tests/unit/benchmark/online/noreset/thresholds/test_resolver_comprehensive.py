"""
Tests for resolver comprehensive.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

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
from tests.support.algorithms import CountingAlgorithm, CountingAlgorithmConfig, CountingAlgorithmState
from tests.support.providers import make_univariate_labeled


def _make_run(values: list[float]) -> SingleRun[OnlineDetectionTrace[CountingAlgorithmState], object]:
    trace = OnlineDetectionTrace(
        detector_description=ChangePointDetectorDescription(name="detector"),
        detected_change_points=[],
        threshold=None,
        processing_time=np.zeros(len(values), dtype=np.float64),
        detection_function=np.asarray(values, dtype=np.float64),
        algorithm_states=[CountingAlgorithmState() for _ in values],
    )
    return SingleRun(trace=trace, provider=make_univariate_labeled())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# auto_pick_thresholds - swap path
# ---------------------------------------------------------------------------


class TestAutoPickThresholdsSwap:
    def test_maintains_min_le_max(self) -> None:
        """Result always satisfies min_threshold <= max_threshold."""
        runs = [_make_run([0.0, 1.0])]
        result = auto_pick_thresholds(
            runs=runs,
            make_recall=lambda _: lambda t: 1.0 if t < 0.6 else 0.0,
            make_precision=lambda _: lambda t: 0.0 if t < 0.3 else 1.0,
            config=ThresholdAutoTuneConfig(threshold_count=5, bs_error=1e-4, t_min=0.0, t_max=1.0),
        )
        assert result.min_threshold <= result.max_threshold


# ---------------------------------------------------------------------------
# resolve_arl_thresholds - manual range with multiple thresholds
# ---------------------------------------------------------------------------


class TestResolveArlThresholdsManual:
    def test_manual_range_with_multiple_thresholds_uses_linspace(self) -> None:
        """For manual range with threshold_count > 1, linspace from
        detection value extremes is used."""
        entry = OnlineNoResetBenchmarkEntry(
            algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
            thresholds=ManualThresholdsRange(_values=(1.0, 2.0, 3.0)),
        )
        resolver = NoResetThresholdResolver()

        resolved = resolver.resolve_arl_thresholds(entry, [_make_run([2.0, 6.0])], None)

        assert len(resolved) == 3
        assert resolved[0] == 2.0
        assert resolved[-1] == 6.0

    def test_auto_range_with_multiple_thresholds_uses_count(self) -> None:
        """For auto range with count > 1, linspace from detection value
        extremes with the configured count is used."""
        entry = OnlineNoResetBenchmarkEntry(
            algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
            thresholds=AutoThresholdsRange(count=4),
        )
        resolver = NoResetThresholdResolver()

        resolved = resolver.resolve_arl_thresholds(entry, [_make_run([0.0, 10.0])], None)

        assert len(resolved) == 4
        assert resolved[0] == 0.0
        assert resolved[-1] == 10.0

    def test_manual_empty_thresholds_range_raises(self) -> None:
        """Empty manual thresholds_range with no explicit thresholds raises."""
        entry = OnlineNoResetBenchmarkEntry(
            algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
            thresholds=ManualThresholdsRange(_values=()),
        )
        resolver = NoResetThresholdResolver()

        with pytest.raises(ValueError, match="empty thresholds_range"):
            resolver.resolve_arl_thresholds(entry, [_make_run([1.0])], None)

    def test_manual_range_with_single_threshold_count(self) -> None:
        """Manual range with threshold_count == 1 returns midpoint."""
        entry = OnlineNoResetBenchmarkEntry(
            algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
            thresholds=ManualThresholdsRange(_values=(0.0,)),
        )
        resolver = NoResetThresholdResolver()

        resolved = resolver.resolve_arl_thresholds(entry, [_make_run([2.0, 6.0])], None)

        assert resolved == [4.0]

    def test_auto_range_multiple_runs_uses_global_min_max(self) -> None:
        """With multiple runs, min and max are taken across all runs."""
        entry = OnlineNoResetBenchmarkEntry(
            algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
            thresholds=AutoThresholdsRange(count=3),
        )
        resolver = NoResetThresholdResolver()

        resolved = resolver.resolve_arl_thresholds(
            entry,
            [_make_run([1.0, 5.0]), _make_run([3.0, 9.0])],
            None,
        )

        assert len(resolved) == 3
        assert resolved[0] == 1.0
        assert resolved[-1] == 9.0


# ---------------------------------------------------------------------------
# resolve_arl_thresholds - explicit thresholds
# ---------------------------------------------------------------------------


class TestResolveArlThresholdsExplicit:
    def test_explicit_thresholds_are_used_directly(self) -> None:
        entry = OnlineNoResetBenchmarkEntry(
            algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
            thresholds=AutoThresholdsRange(count=4),
        )
        resolver = NoResetThresholdResolver()

        resolved = resolver.resolve_arl_thresholds(entry, [_make_run([1.0, 2.0])], [2.0, 5.0])

        assert resolved == [2.0, 5.0]

    def test_empty_explicit_thresholds_falls_through_to_infer(self) -> None:
        """Empty explicit thresholds list should fall through to inference."""
        entry = OnlineNoResetBenchmarkEntry(
            algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
            thresholds=AutoThresholdsRange(count=3),
        )
        resolver = NoResetThresholdResolver()

        resolved = resolver.resolve_arl_thresholds(entry, [_make_run([2.0, 6.0])], [])

        assert len(resolved) == 3
        assert resolved[0] == 2.0
        assert resolved[-1] == 6.0
