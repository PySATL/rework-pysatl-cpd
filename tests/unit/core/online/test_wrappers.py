# -*- coding: ascii -*-
"""
Tests for wrappers.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math
from dataclasses import dataclass

from pysatl_cpd.core.online import (
    BatchingOnlineAlgorithmWrapper,
    BatchReducer,
    OnlineResetDetector,
    SkippingCondition,
    SkippingOnlineAlgorithmWrapper,
)
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithm, OnlineAlgorithmConfiguration, OnlineAlgorithmState
from pysatl_cpd.typedefs import stable_hash
from tests.support.core import (
    MockAlgorithmConfiguration,
    MockDataProvider,
    MockOnlineAlgorithm,
)


@dataclass(kw_only=True, frozen=True)
class NoArgRecreateConfiguration(OnlineAlgorithmConfiguration):
    return_value: float = 1.0


@dataclass(kw_only=True, frozen=True)
class NoArgRecreateState(OnlineAlgorithmState):
    call_count: int = 0


class NoArgRecreateAlgorithm(OnlineAlgorithm[float, NoArgRecreateConfiguration, NoArgRecreateState]):
    def __init__(self, configuration: NoArgRecreateConfiguration | None = None) -> None:
        self._configuration = configuration if configuration is not None else NoArgRecreateConfiguration()
        self._state = NoArgRecreateState()

    @property
    def name(self) -> str:
        return "NoArgRecreateAlgorithm"

    @property
    def configuration(self) -> NoArgRecreateConfiguration:
        return self._configuration

    @property
    def state(self) -> NoArgRecreateState:
        return self._state

    def process(self, observation: float) -> float:
        self._state = NoArgRecreateState(call_count=self._state.call_count + 1)
        return self._configuration.return_value

    def reset(self) -> None:
        self._state = NoArgRecreateState()

    def recreate(self) -> NoArgRecreateAlgorithm:
        return type(self)(self._configuration)


class TestSkippingCondition:
    def test_rejects_empty_name(self) -> None:
        try:
            SkippingCondition[float](name="", condition=lambda _: True)
            raise AssertionError("Expected ValueError")
        except ValueError as exc:
            assert "non-empty" in str(exc)

    def test_hash_depends_only_on_name(self) -> None:
        left = SkippingCondition[float](name="nan", condition=lambda _: True)
        right = SkippingCondition[float](name="nan", condition=lambda _: False)

        assert hash(left) == hash(right) == stable_hash((left.__class__.__module__, left.__class__.__qualname__, "nan"))


class TestBatchReducer:
    def test_rejects_empty_name(self) -> None:
        try:
            BatchReducer[float, float](name="", reducer=sum)
            raise AssertionError("Expected ValueError")
        except ValueError as exc:
            assert "non-empty" in str(exc)

    def test_hash_depends_only_on_name(self) -> None:
        left = BatchReducer[float, float](name="sum", reducer=sum)
        right = BatchReducer[float, float](name="sum", reducer=max)

        assert hash(left) == hash(right) == stable_hash((left.__class__.__module__, left.__class__.__qualname__, "sum"))


class TestSkippingOnlineAlgorithmWrapper:
    def test_skips_matching_observations_and_repeats_last_value(self) -> None:
        algorithm = MockOnlineAlgorithm(MockAlgorithmConfiguration(return_value=1.5))
        wrapper = SkippingOnlineAlgorithmWrapper(
            algorithm,
            skipping_condition=SkippingCondition(name="nan", condition=lambda value: math.isnan(value)),
        )

        skipped_result = wrapper.process(float("nan"))
        processed_result = wrapper.process(2.0)
        repeated_result = wrapper.process(float("nan"))

        assert math.isnan(skipped_result)
        assert processed_result == 1.5
        assert repeated_result == 1.5
        assert algorithm.state.call_count == 1

    def test_name_includes_skipping_suffix(self) -> None:
        wrapper = SkippingOnlineAlgorithmWrapper(
            MockOnlineAlgorithm(),
            skipping_condition=SkippingCondition(name="nan", condition=lambda value: math.isnan(value)),
        )

        assert wrapper.name == "MockOnlineAlgorithm{skip[on=nan]}"

    def test_reset_clears_last_value(self) -> None:
        wrapper = SkippingOnlineAlgorithmWrapper(
            MockOnlineAlgorithm(MockAlgorithmConfiguration(return_value=2.0)),
            skipping_condition=SkippingCondition(name="neg", condition=lambda value: value < 0),
        )

        assert wrapper.process(1.0) == 2.0
        wrapper.reset()
        reset_result = wrapper.process(-1.0)

        assert math.isnan(reset_result)
        assert wrapper.state.call_count == 0

    def test_recreate_returns_fresh_wrapper(self) -> None:
        wrapper = SkippingOnlineAlgorithmWrapper(
            MockOnlineAlgorithm(MockAlgorithmConfiguration(return_value=2.0)),
            skipping_condition=SkippingCondition(name="neg", condition=lambda value: value < 0),
        )
        wrapper.process(1.0)

        recreated = wrapper.recreate()

        assert recreated is not wrapper
        assert recreated.name == wrapper.name
        assert recreated.state.call_count == 0
        assert math.isnan(recreated.process(-1.0))


class TestBatchingOnlineAlgorithmWrapper:
    def test_requires_positive_batch_size(self) -> None:
        try:
            BatchingOnlineAlgorithmWrapper(
                MockOnlineAlgorithm(),
                batch_size=0,
                reducer=BatchReducer(name="sum", reducer=sum),
            )
            raise AssertionError("Expected ValueError")
        except ValueError as exc:
            assert "positive" in str(exc)

    def test_batches_observations_and_repeats_last_value_between_emissions(self) -> None:
        algorithm = MockOnlineAlgorithm(MockAlgorithmConfiguration(return_value=0.7))
        wrapper = BatchingOnlineAlgorithmWrapper(
            algorithm,
            batch_size=3,
            reducer=BatchReducer(name="sum", reducer=sum),
        )

        first = wrapper.process(1.0)
        second = wrapper.process(2.0)
        third = wrapper.process(3.0)
        fourth = wrapper.process(4.0)

        assert math.isnan(first)
        assert math.isnan(second)
        assert third == 0.7
        assert fourth == 0.7
        assert algorithm.state.call_count == 1

    def test_discards_trailing_incomplete_batch(self) -> None:
        algorithm = MockOnlineAlgorithm(MockAlgorithmConfiguration(return_value=0.9))
        wrapper = BatchingOnlineAlgorithmWrapper(
            algorithm,
            batch_size=3,
            reducer=BatchReducer(name="sum", reducer=sum),
        )

        results = [wrapper.process(value) for value in [1.0, 2.0, 3.0, 4.0, 5.0]]

        assert results[2] == 0.9
        assert results[3] == 0.9
        assert results[4] == 0.9
        assert algorithm.state.call_count == 1

    def test_name_includes_batching_suffix(self) -> None:
        wrapper = BatchingOnlineAlgorithmWrapper(
            MockOnlineAlgorithm(),
            batch_size=3,
            reducer=BatchReducer(name="pca", reducer=sum),
        )

        assert wrapper.name == "MockOnlineAlgorithm{batch[size=3, reduce=pca]}"

    def test_reset_clears_buffer_and_last_value(self) -> None:
        wrapper = BatchingOnlineAlgorithmWrapper(
            MockOnlineAlgorithm(MockAlgorithmConfiguration(return_value=2.0)),
            batch_size=2,
            reducer=BatchReducer(name="sum", reducer=sum),
        )
        wrapper.process(1.0)
        wrapper.reset()

        reset_result = wrapper.process(2.0)

        assert math.isnan(reset_result)
        assert wrapper.state.call_count == 0

    def test_recreate_returns_fresh_wrapper(self) -> None:
        wrapper = BatchingOnlineAlgorithmWrapper(
            MockOnlineAlgorithm(MockAlgorithmConfiguration(return_value=2.0)),
            batch_size=2,
            reducer=BatchReducer(name="sum", reducer=sum),
        )
        wrapper.process(1.0)
        wrapper.process(2.0)

        recreated = wrapper.recreate()

        assert recreated is not wrapper
        assert recreated.name == wrapper.name
        assert recreated.state.call_count == 0
        assert math.isnan(recreated.process(1.0))


def test_wrappers_recreate_terminal_algorithm_with_no_recreate_parameters() -> None:
    skipping = SkippingOnlineAlgorithmWrapper(
        NoArgRecreateAlgorithm(),
        skipping_condition=SkippingCondition(name="neg", condition=lambda value: value < 0),
    )
    batching = BatchingOnlineAlgorithmWrapper(
        NoArgRecreateAlgorithm(),
        batch_size=2,
        reducer=BatchReducer(name="sum", reducer=sum),
    )

    recreated_skipping = skipping.recreate()
    recreated_batching = batching.recreate()

    assert recreated_skipping is not skipping
    assert isinstance(recreated_skipping._algorithm, NoArgRecreateAlgorithm)
    assert recreated_skipping.state.call_count == 0
    assert recreated_batching is not batching
    assert isinstance(recreated_batching._algorithm, NoArgRecreateAlgorithm)
    assert recreated_batching.state.call_count == 0


class TestWrapperNameComposition:
    def test_batch_wrapping_skip_appends_suffix_outermost_last(self) -> None:
        wrapped = BatchingOnlineAlgorithmWrapper(
            SkippingOnlineAlgorithmWrapper(
                MockOnlineAlgorithm(),
                skipping_condition=SkippingCondition(name="nan", condition=lambda value: math.isnan(value)),
            ),
            batch_size=3,
            reducer=BatchReducer(name="pca", reducer=sum),
        )

        assert wrapped.name == "MockOnlineAlgorithm{skip[on=nan]}{batch[size=3, reduce=pca]}"


class TestWrappersWithOnlineResetDetector:
    def test_skipping_wrapper_keeps_trace_length_and_repeats_cached_values(self) -> None:
        detector = OnlineResetDetector(
            SkippingOnlineAlgorithmWrapper(
                MockOnlineAlgorithm(MockAlgorithmConfiguration(return_value=0.5, increment=0.5)),
                skipping_condition=SkippingCondition(name="neg", condition=lambda value: value < 0),
            ),
            threshold=float("nan"),
            collect_states=True,
        )

        trace = detector.detect(MockDataProvider([1.0, -1.0, 2.0, -2.0, 3.0]))

        assert len(trace.detection_function) == 5
        assert trace.detection_function.tolist() == [0.5, 0.5, 1.0, 1.0, 1.5]
        assert [state.call_count if state is not None else None for state in trace.algorithm_states] == [1, 1, 2, 2, 3]

    def test_batching_wrapper_keeps_trace_length_and_duplicates_states(self) -> None:
        detector = OnlineResetDetector(
            BatchingOnlineAlgorithmWrapper(
                MockOnlineAlgorithm(MockAlgorithmConfiguration(return_value=0.5, increment=0.5)),
                batch_size=2,
                reducer=BatchReducer(name="sum", reducer=sum),
            ),
            threshold=float("nan"),
            collect_states=True,
        )

        trace = detector.detect(MockDataProvider([1.0, 2.0, 3.0, 4.0, 5.0]))

        assert len(trace.detection_function) == 5
        assert math.isnan(trace.detection_function[0])
        assert trace.detection_function[1:].tolist() == [0.5, 0.5, 1.0, 1.0]
        assert [state.call_count if state is not None else None for state in trace.algorithm_states] == [0, 1, 1, 2, 2]
