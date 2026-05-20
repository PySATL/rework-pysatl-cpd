# -*- coding: ascii -*-
"""
Tests for registry parallel.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.core.online import (
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
    OnlineDetectionTrace,
    OnlineResetDetector,
)
from pysatl_cpd.data import LabeledData, TimeseriesAnnotation
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


def _providers() -> Sequence[LabeledData[float, TimeseriesAnnotation]]:
    return cast(
        Sequence[LabeledData[float, TimeseriesAnnotation]],
        [
            MockDataProvider([1.0, 2.0, 3.0], name="series-a"),
            MockDataProvider([1.0, 2.0, 3.0], name="series-b"),
        ],
    )


def test_parallel_update_clones_detector_and_merges_results_in_parent() -> None:
    detector: OnlineResetDetector[float, TimeseriesAnnotation, ConstantConfig, CountingState] = OnlineResetDetector(
        ConstantAlgorithm(ConstantConfig(value=0.5)),
        threshold=0.1,
    )
    registry: BenchmarkRegistry[float, OnlineDetectionTrace[Any]] = BenchmarkRegistry()

    registry.update(detector, _providers(), n_jobs=2, backend="threading")

    assert len(registry) == 2
    assert detector.algorithm.state.call_count == 0
    assert all(len(run.trace.detected_change_points) == 3 for run in registry.values())
