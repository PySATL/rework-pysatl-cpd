# -*- coding: ascii -*-
"""
Tests for algorithms.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass

import numpy as np

from pysatl_cpd.core.change_point_detector import ChangePointDetector, ChangePointDetectorDescription
from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.core.online import (
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
    OnlineDetectionTrace,
    OnlineDetector,
)
from pysatl_cpd.data import DataProvider
from pysatl_cpd.data.providers.transformers.base import IDataTransformer
from pysatl_cpd.data.typedefs import TimeseriesAnnotation
from pysatl_cpd.typedefs import Number, stable_hash


@dataclass(kw_only=True, frozen=True)
class CountingAlgorithmState(OnlineAlgorithmState):
    """State carrying a call counter and running total."""

    call_count: int = 0
    running_total: float = 0.0

    def __hash__(self) -> int:
        return stable_hash(
            (
                type(self).__module__,
                type(self).__qualname__,
                self.is_in_learning_period,
                self.call_count,
                self.running_total,
            )
        )


@dataclass(kw_only=True, frozen=True)
class CountingAlgorithmConfig(OnlineAlgorithmConfiguration):
    """Configuration for the counting algorithm."""

    threshold: float = 3.0

    def __hash__(self) -> int:
        return stable_hash((type(self).__module__, type(self).__qualname__, self.learning_period_size, self.threshold))


class CountingAlgorithm(OnlineAlgorithm[float, CountingAlgorithmConfig, CountingAlgorithmState]):
    """Simple deterministic online algorithm."""

    def __init__(self, config: CountingAlgorithmConfig) -> None:
        self._config = config
        self._state = CountingAlgorithmState()

    @property
    def configuration(self) -> CountingAlgorithmConfig:
        return self._config

    @property
    def state(self) -> CountingAlgorithmState:
        return self._state

    def process(self, observation: float) -> Number:
        total = self._state.running_total + float(observation)
        self._state = CountingAlgorithmState(call_count=self._state.call_count + 1, running_total=total)
        return total

    def reset(self) -> None:
        self._state = CountingAlgorithmState()

    def recreate(self) -> CountingAlgorithm:
        return type(self)(self.configuration)


class IdentityTransformer(
    IDataTransformer[DataProvider[float, TimeseriesAnnotation], DataProvider[float, TimeseriesAnnotation]]
):
    """Transformer that records the last provider and returns it."""

    def __init__(self, annotation: str = "identity") -> None:
        self._annotation = annotation
        self.last_provider: DataProvider[float, TimeseriesAnnotation] | None = None

    @property
    def annotation(self) -> str:
        return self._annotation

    def transform(
        self, provider: DataProvider[float, TimeseriesAnnotation]
    ) -> DataProvider[float, TimeseriesAnnotation]:
        self.last_provider = provider
        return provider


class RecordingDetector(ChangePointDetector[float]):
    """Detector that records which provider reached detection."""

    def __init__(self, *, data_transformer: IDataTransformer | None = None) -> None:
        super().__init__(data_transformer=data_transformer)
        self.last_provider: DataProvider[float, TimeseriesAnnotation] | None = None

    @property
    def description(self) -> ChangePointDetectorDescription:
        return ChangePointDetectorDescription(name="recording_detector")

    def _detect(self, data: DataProvider[float, TimeseriesAnnotation]) -> DetectionTrace:
        self.last_provider = data
        return DetectionTrace(detected_change_points=[1], detector_description=self.description)

    def clone(self) -> RecordingDetector:
        return type(self)(data_transformer=self.data_transformer)


class RecordingOnlineDetector(OnlineDetector[float, CountingAlgorithmConfig, CountingAlgorithmState]):
    """Online detector that returns a deterministic trace."""

    def __init__(self, algorithm: CountingAlgorithm, *, data_transformer: IDataTransformer | None = None) -> None:
        super().__init__(algorithm, data_transformer=data_transformer)
        self.last_provider: DataProvider[float, TimeseriesAnnotation] | None = None

    @property
    def description(self) -> ChangePointDetectorDescription:
        return ChangePointDetectorDescription(name="recording_online_detector")

    def _detect(self, data: DataProvider[float, TimeseriesAnnotation]) -> OnlineDetectionTrace[CountingAlgorithmState]:
        self.last_provider = data
        scores = np.asarray([float(value) for value in data], dtype=np.float64)
        return OnlineDetectionTrace(
            detector_description=self.description,
            detected_change_points=[
                idx for idx, value in enumerate(scores) if value >= self.algorithm.configuration.threshold
            ],
            threshold=self.algorithm.configuration.threshold,
            processing_time=np.zeros(len(scores), dtype=np.float64),
            detection_function=scores,
            algorithm_states=[self.algorithm.state for _ in range(len(scores))],
        )

    def clone(self) -> RecordingOnlineDetector:
        return type(self)(self.algorithm.recreate(), data_transformer=self.data_transformer)
