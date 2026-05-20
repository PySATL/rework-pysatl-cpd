# -*- coding: ascii -*-
"""
Tests for core.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Self, cast

import numpy as np

from pysatl_cpd.core.change_point_detector import ChangePointDetector, ChangePointDetectorDescription
from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.core.online import OnlineDetector
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithm, OnlineAlgorithmConfiguration, OnlineAlgorithmState
from pysatl_cpd.data.providers.data_provider import DataProvider
from pysatl_cpd.data.typedefs import TimeseriesAnnotation
from pysatl_cpd.typedefs import Number, stable_hash


def assert_nested_equal(actual, expected) -> None:
    """Assert equality for nested mappings/lists containing numeric values."""

    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, expected_value in expected.items():
            assert_nested_equal(actual[key], expected_value)
        return
    if isinstance(expected, list):
        np.testing.assert_allclose(np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64))
        return
    if isinstance(expected, float):
        assert np.isclose(actual, expected)
        return
    assert actual == expected


class MockDataProvider(DataProvider[float, TimeseriesAnnotation]):
    """Small concrete provider for core tests."""

    def __init__(self, data: list[float] | None = None, name: str = "mock_provider") -> None:
        self._data = data if data is not None else [1.0, 2.0, 3.0, 4.0, 5.0]
        self._annotation = TimeseriesAnnotation(name=name)

    @property
    def annotation(self) -> TimeseriesAnnotation:
        return self._annotation

    def __iter__(self) -> Iterator[float]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def cut[AnnotationSliceT: TimeseriesAnnotation](
        self,
        start: int,
        stop: int,
        *,
        annotation: AnnotationSliceT,
        name: str | None = None,
    ) -> DataProvider[float, AnnotationSliceT]:
        return cast(DataProvider[float, AnnotationSliceT], type(self)(self._data[start : stop + 1], annotation.name))

    @classmethod
    def merge[MergedAnnotationT: TimeseriesAnnotation](
        cls,
        providers: Sequence[Self],
        annotation_builder: Callable[[Sequence[TimeseriesAnnotation]], MergedAnnotationT] | None = None,
    ) -> DataProvider[float, MergedAnnotationT]:
        if not providers:
            raise ValueError("merge requires at least one provider")
        merged_data = [value for provider in providers for value in provider]
        annotation = (
            annotation_builder([provider.annotation for provider in providers])
            if annotation_builder is not None
            else TimeseriesAnnotation(name="merged")
        )
        return cast(DataProvider[float, MergedAnnotationT], cls(merged_data, annotation.name))


class MockUnivariateDataProvider(DataProvider[np.float64, TimeseriesAnnotation]):
    """Numpy-backed univariate provider for tests that need scalar numpy values."""

    def __init__(self, data: list[float] | None = None, name: str = "mock_univariate") -> None:
        self._data = np.asarray(data if data is not None else [1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        self._annotation = TimeseriesAnnotation(name=name)

    @property
    def annotation(self) -> TimeseriesAnnotation:
        return self._annotation

    def __iter__(self) -> Iterator[np.float64]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def cut[AnnotationSliceT: TimeseriesAnnotation](
        self,
        start: int,
        stop: int,
        *,
        annotation: AnnotationSliceT,
        name: str | None = None,
    ) -> DataProvider[np.float64, AnnotationSliceT]:
        return cast(
            DataProvider[np.float64, AnnotationSliceT],
            type(self)(self._data[start : stop + 1].tolist(), annotation.name),
        )

    @classmethod
    def merge[MergedAnnotationT: TimeseriesAnnotation](
        cls,
        providers: Sequence[Self],
        annotation_builder: Callable[[Sequence[TimeseriesAnnotation]], MergedAnnotationT] | None = None,
    ) -> DataProvider[np.float64, MergedAnnotationT]:
        if not providers:
            raise ValueError("merge requires at least one provider")
        merged_data = np.concatenate([np.asarray(list(provider), dtype=np.float64) for provider in providers])
        annotation = (
            annotation_builder([provider.annotation for provider in providers])
            if annotation_builder is not None
            else TimeseriesAnnotation(name="merged")
        )
        return cast(DataProvider[np.float64, MergedAnnotationT], cls(merged_data.tolist(), annotation.name))


@dataclass(kw_only=True, frozen=True)
class MockAlgorithmState(OnlineAlgorithmState):
    """State for a fixed-output online algorithm."""

    call_count: int = 0

    def __hash__(self) -> int:
        return stable_hash(
            (type(self).__module__, type(self).__qualname__, self.is_in_learning_period, self.call_count)
        )


@dataclass(kw_only=True, frozen=True)
class MockAlgorithmConfiguration(OnlineAlgorithmConfiguration):
    """Configuration for a fixed-output online algorithm."""

    return_value: float = 0.5
    increment: float = 0.0

    def __hash__(self) -> int:
        return stable_hash(
            (
                type(self).__module__,
                type(self).__qualname__,
                self.learning_period_size,
                self.return_value,
                self.increment,
            )
        )


class MockOnlineAlgorithm(OnlineAlgorithm[float, MockAlgorithmConfiguration, MockAlgorithmState]):
    """Online algorithm that returns fixed or incrementing values."""

    def __init__(self, config: MockAlgorithmConfiguration | None = None) -> None:
        self._config = config if config is not None else MockAlgorithmConfiguration()
        self._state = MockAlgorithmState()
        self._call_count = 0

    @property
    def name(self) -> str:
        return "MockOnlineAlgorithm"

    @property
    def configuration(self) -> MockAlgorithmConfiguration:
        return self._config

    @property
    def state(self) -> MockAlgorithmState:
        return self._state

    def process(self, observation: float) -> Number:
        self._call_count += 1
        self._state = MockAlgorithmState(call_count=self._call_count)
        if self._config.increment != 0:
            return self._config.return_value + (self._call_count - 1) * self._config.increment
        return self._config.return_value

    def reset(self) -> None:
        self._call_count = 0
        self._state = MockAlgorithmState()

    @classmethod
    def recreate(
        cls, configuration: MockAlgorithmConfiguration, state: MockAlgorithmState | None = None
    ) -> MockOnlineAlgorithm:
        algorithm = cls(configuration)
        if state is not None:
            algorithm._call_count = state.call_count
            algorithm._state = state
        return algorithm


@dataclass(kw_only=True, frozen=True)
class SimpleAlgorithmState(OnlineAlgorithmState):
    """State for simple source-class tests."""

    value: float = 0.0

    def __hash__(self) -> int:
        return stable_hash((type(self).__module__, type(self).__qualname__, self.is_in_learning_period, self.value))


@dataclass(kw_only=True, frozen=True)
class SimpleAlgorithmConfig(OnlineAlgorithmConfiguration):
    """Configuration for simple source-class tests."""

    threshold: float = 0.5

    def __hash__(self) -> int:
        return stable_hash((type(self).__module__, type(self).__qualname__, self.learning_period_size, self.threshold))


class SimpleAlgorithm(OnlineAlgorithm[float, SimpleAlgorithmConfig, SimpleAlgorithmState]):
    """Minimal concrete online algorithm."""

    def __init__(self, config: SimpleAlgorithmConfig) -> None:
        self._config = config
        self._state = SimpleAlgorithmState()

    @property
    def name(self) -> str:
        return "SimpleAlgorithm"

    @property
    def configuration(self) -> SimpleAlgorithmConfig:
        return self._config

    @property
    def state(self) -> SimpleAlgorithmState:
        return self._state

    def process(self, observation: float) -> Number:
        return 0.5

    def reset(self) -> None:
        self._state = SimpleAlgorithmState()

    @classmethod
    def recreate(
        cls, configuration: SimpleAlgorithmConfig, state: SimpleAlgorithmState | None = None
    ) -> SimpleAlgorithm:
        algorithm = cls(configuration)
        if state is not None:
            algorithm._state = state
        return algorithm


class MockChangePointDetector(ChangePointDetector[float]):
    """Simple detector for ChangePointDetector tests."""

    @property
    def description(self) -> ChangePointDetectorDescription:
        return ChangePointDetectorDescription(name="MockChangePointDetector")

    def _detect(self, data: DataProvider[float, TimeseriesAnnotation]) -> DetectionTrace:
        return DetectionTrace(detected_change_points=[], detector_description=self.description)

    def clone(self) -> Self:
        return type(self)()


class ConcreteOnlineDetector(OnlineDetector[float, OnlineAlgorithmConfiguration, OnlineAlgorithmState]):
    """Concrete online detector for source-class tests."""

    def __init__(self, algorithm: OnlineAlgorithm) -> None:
        super().__init__(algorithm)

    @property
    def description(self) -> ChangePointDetectorDescription:
        return ChangePointDetectorDescription(name="ConcreteOnlineDetector")

    def _detect(self, data: DataProvider[float, TimeseriesAnnotation]) -> OnlineDetectionTrace:
        return OnlineDetectionTrace(
            detector_description=self.description,
            detection_function=[],
            processing_time=[],
            algorithm_states=[],
        )

    def clone(self) -> Self:
        return type(self)(type(self.algorithm).recreate(self.algorithm.configuration))
