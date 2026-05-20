# -*- coding: ascii -*-
"""
Tests for benchmark.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import pytest

from pysatl_cpd.analysis.metrics.abstracts.imultiple_run_metric import IMultipleRunMetric
from pysatl_cpd.benchmark.online.reset import OnlineResetBenchmark, OnlineResetBenchmarkEntry
from pysatl_cpd.benchmark.online.reset.scenarios import OnlineResetWholeTimeseriesMetricScenario
from pysatl_cpd.benchmark.registry import DEFAULT_JOB_PARALLEL_BACKEND, BenchmarkRegistry
from pysatl_cpd.core.change_point_detector import ChangePointDetector
from pysatl_cpd.core.online import (
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
    OnlineDetectionTrace,
    OnlineResetDetector,
)
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data import LabeledData, TimeseriesAnnotation
from pysatl_cpd.data.dataset import Dataset
from pysatl_cpd.typedefs import Number
from tests.support.core import MockDataProvider


@dataclass(kw_only=True, frozen=True)
class CountingState(OnlineAlgorithmState):
    call_count: int = 0


@dataclass(kw_only=True, frozen=True)
class ConstantConfig(OnlineAlgorithmConfiguration):
    value: float = 0.5


class ConstantAlgorithm(OnlineAlgorithm[float, ConstantConfig, CountingState]):
    def __init__(self, config: ConstantConfig) -> None:
        self._config = config
        self._state = CountingState()

    @property
    def name(self) -> str:
        return "ConstantAlgorithm"

    @property
    def configuration(self) -> ConstantConfig:
        return self._config

    @property
    def state(self) -> CountingState:
        return self._state

    def process(self, observation: float) -> Number:
        self._state = CountingState(call_count=self._state.call_count + 1)
        return self._config.value

    def reset(self) -> None:
        self._state = CountingState()

    def recreate(self) -> ConstantAlgorithm:
        return ConstantAlgorithm(self._config)


class DetectedChangePointsMetric(
    IMultipleRunMetric[OnlineDetectionTrace[Any], LabeledData[float, TimeseriesAnnotation], int],
):
    def evaluate(
        self,
        runs: Sequence[SingleRun[OnlineDetectionTrace[Any], LabeledData[float, TimeseriesAnnotation]]],
    ) -> int:
        return sum(len(run.trace.detected_change_points) for run in runs)


def test_get_metrics_for_benchmarks_cloned_reset_detectors_on_whole_timeseries() -> None:
    dataset = Dataset(
        cast(
            Sequence[LabeledData[float, TimeseriesAnnotation]],
            [MockDataProvider([1.0, 2.0, 3.0], name="series-a")],
        ),
    )
    registry: BenchmarkRegistry[float, OnlineDetectionTrace[Any]] = BenchmarkRegistry()
    benchmark = OnlineResetBenchmark(dataset, registry)
    detector: OnlineResetDetector[float, TimeseriesAnnotation, ConstantConfig, CountingState] = OnlineResetDetector(
        ConstantAlgorithm(ConstantConfig(value=0.5)),
        threshold=0.1,
    )
    entry = OnlineResetBenchmarkEntry(detector)

    result = benchmark.get_metrics_for([entry], DetectedChangePointsMetric(), n_jobs=2, backend="threading")

    assert result == {entry.description: 3}
    assert detector.algorithm.state.call_count == 0


def test_n_jobs_property_is_used_as_default(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = Dataset(
        cast(
            Sequence[LabeledData[float, TimeseriesAnnotation]],
            [MockDataProvider([1.0, 2.0, 3.0], name="series-a")],
        ),
    )
    registry: BenchmarkRegistry[float, OnlineDetectionTrace[Any]] = BenchmarkRegistry()
    benchmark = OnlineResetBenchmark(dataset, registry, n_jobs=2)
    detector: OnlineResetDetector[float, TimeseriesAnnotation, ConstantConfig, CountingState] = OnlineResetDetector(
        ConstantAlgorithm(ConstantConfig(value=0.5)),
        threshold=0.1,
    )
    scenario = OnlineResetWholeTimeseriesMetricScenario(
        [OnlineResetBenchmarkEntry(detector)],
        DetectedChangePointsMetric(),
    )
    captured: dict[str, int] = {}

    def update_spy(
        detector: ChangePointDetector[float],
        providers: Sequence[LabeledData[float, TimeseriesAnnotation]],
        *,
        force_recompute: bool = False,
        n_jobs: int = 1,
        backend: str = DEFAULT_JOB_PARALLEL_BACKEND,
    ) -> None:
        captured["n_jobs"] = n_jobs

    monkeypatch.setattr(registry, "update", update_spy)

    benchmark.run_scenario(scenario)

    assert captured["n_jobs"] == 2


def test_run_scenario_n_jobs_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = Dataset(
        cast(
            Sequence[LabeledData[float, TimeseriesAnnotation]],
            [MockDataProvider([1.0, 2.0, 3.0], name="series-a")],
        ),
    )
    registry: BenchmarkRegistry[float, OnlineDetectionTrace[Any]] = BenchmarkRegistry()
    benchmark = OnlineResetBenchmark(dataset, registry, n_jobs=2)
    detector: OnlineResetDetector[float, TimeseriesAnnotation, ConstantConfig, CountingState] = OnlineResetDetector(
        ConstantAlgorithm(ConstantConfig(value=0.5)),
        threshold=0.1,
    )
    captured: dict[str, int] = {}

    def update_spy(
        detector: ChangePointDetector[float],
        providers: Sequence[LabeledData[float, TimeseriesAnnotation]],
        *,
        force_recompute: bool = False,
        n_jobs: int = 1,
        backend: str = DEFAULT_JOB_PARALLEL_BACKEND,
    ) -> None:
        captured["n_jobs"] = n_jobs

    scenario = OnlineResetWholeTimeseriesMetricScenario(
        [OnlineResetBenchmarkEntry(detector)],
        DetectedChangePointsMetric(),
    )
    monkeypatch.setattr(registry, "update", update_spy)

    benchmark.run_scenario(scenario, n_jobs=1)

    assert captured["n_jobs"] == 1


def test_n_jobs_rejects_zero() -> None:
    dataset = Dataset(
        cast(
            Sequence[LabeledData[float, TimeseriesAnnotation]],
            [MockDataProvider([1.0, 2.0, 3.0], name="series-a")],
        ),
    )
    registry: BenchmarkRegistry[float, OnlineDetectionTrace[Any]] = BenchmarkRegistry()

    with pytest.raises(ValueError, match="n_jobs must be non-zero"):
        OnlineResetBenchmark(dataset, registry, n_jobs=0)
