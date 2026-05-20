# -*- coding: ascii -*-
"""
Tests for online reset detector.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.core.online import OnlineResetDetector
from tests.support.core import (
    MockAlgorithmConfiguration,
    MockDataProvider,
    MockOnlineAlgorithm,
)


@pytest.fixture
def simple_algorithm() -> MockOnlineAlgorithm:
    """Algorithm that always returns 0.3."""
    config = MockAlgorithmConfiguration(return_value=0.3)
    return MockOnlineAlgorithm(config)


@pytest.fixture
def counting_algorithm() -> MockOnlineAlgorithm:
    """Algorithm that increments return value each call."""
    config = MockAlgorithmConfiguration(return_value=0.0, increment=0.1)
    return MockOnlineAlgorithm(config)


@pytest.fixture
def data_provider() -> MockDataProvider:
    """Provide a simple data provider with 5 observations."""
    return MockDataProvider([1.0, 2.0, 3.0, 4.0, 5.0])


class TestOnlineResetDetectorInit:
    """Tests for OnlineResetDetector initialization."""

    def test_default_init(self) -> None:
        """Default initialization should work."""
        detector = OnlineResetDetector(MockOnlineAlgorithm(MockAlgorithmConfiguration()))
        assert detector is not None

    def test_valid_skip_period(self) -> None:
        """Non-negative skip_period should be accepted."""
        detector = OnlineResetDetector(MockOnlineAlgorithm(MockAlgorithmConfiguration()), skip_period=5)
        assert detector is not None

    def test_negative_skip_period_raises_value_error(self) -> None:
        """Negative skip_period should raise ValueError."""
        with pytest.raises(ValueError, match="skip_period must be non-negative"):
            OnlineResetDetector(MockOnlineAlgorithm(MockAlgorithmConfiguration()), skip_period=-1)

    def test_valid_max_runlength(self) -> None:
        """Positive max_runlength should be accepted."""
        detector = OnlineResetDetector(MockOnlineAlgorithm(MockAlgorithmConfiguration()), max_runlength=10)
        assert detector is not None

    def test_none_max_runlength(self) -> None:
        """None max_runlength should be accepted."""
        detector = OnlineResetDetector(MockOnlineAlgorithm(MockAlgorithmConfiguration()), max_runlength=None)
        assert detector is not None

    def test_zero_max_runlength_raises_value_error(self) -> None:
        """Zero max_runlength should raise ValueError."""
        with pytest.raises(ValueError, match="max_runlength must be positive"):
            OnlineResetDetector(MockOnlineAlgorithm(MockAlgorithmConfiguration()), max_runlength=0)

    def test_negative_max_runlength_raises_value_error(self) -> None:
        """Negative max_runlength should raise ValueError."""
        with pytest.raises(ValueError, match="max_runlength must be positive"):
            OnlineResetDetector(MockOnlineAlgorithm(MockAlgorithmConfiguration()), max_runlength=-5)


class TestOnlineResetDetectorDetect:
    """Tests for OnlineResetDetector.detect() method."""

    def test_yields_correct_number_of_steps(
        self, simple_algorithm: MockOnlineAlgorithm, data_provider: MockDataProvider
    ) -> None:
        """Should yield one step result per observation."""
        detector = OnlineResetDetector(simple_algorithm)
        trace = detector.detect(data_provider)
        assert len(trace.detection_function) == 5

    def test_step_num_is_zero_based_and_sequential(
        self, simple_algorithm: MockOnlineAlgorithm, data_provider: MockDataProvider
    ) -> None:
        """step_num should start at 0 and increment by 1."""
        detector = OnlineResetDetector(simple_algorithm, threshold=0.1)
        trace = detector.detect(data_provider)
        assert trace.signal_change_points == list(range(len(data_provider)))

    def test_processing_time_is_non_negative(
        self, simple_algorithm: MockOnlineAlgorithm, data_provider: MockDataProvider
    ) -> None:
        """processing_time should be non-negative."""
        detector = OnlineResetDetector(simple_algorithm)
        trace = detector.detect(data_provider)
        for processing_time in trace.processing_time:
            assert processing_time >= 0

    def test_detection_function_stored(
        self, simple_algorithm: MockOnlineAlgorithm, data_provider: MockDataProvider
    ) -> None:
        """detection_function should store the algorithm's return value."""
        detector = OnlineResetDetector(simple_algorithm)
        trace = detector.detect(data_provider)
        for detection_function in trace.detection_function:
            assert detection_function == 0.3

    def test_threshold_nan_no_detections(
        self, simple_algorithm: MockOnlineAlgorithm, data_provider: MockDataProvider
    ) -> None:
        """With threshold=nan, no change points should be detected."""
        detector = OnlineResetDetector(simple_algorithm, threshold=float("nan"))
        trace = detector.detect(data_provider)
        assert trace.detected_change_points == []
        assert trace.signal_change_points == []

    def test_signal_change_point_above_threshold(self, data_provider: MockDataProvider) -> None:
        """Detection should fire when value exceeds threshold."""
        config = MockAlgorithmConfiguration(return_value=0.5)
        algorithm = MockOnlineAlgorithm(config)
        detector = OnlineResetDetector(algorithm, threshold=0.3)
        trace = detector.detect(data_provider)
        assert trace.signal_change_points == list(range(len(data_provider)))

    def test_signal_change_point_first_step(
        self, simple_algorithm: MockOnlineAlgorithm, data_provider: MockDataProvider
    ) -> None:
        """Detection should fire on first step if value exceeds threshold."""
        detector = OnlineResetDetector(simple_algorithm, threshold=0.1)
        trace = detector.detect(data_provider)
        assert trace.signal_change_points[0] == 0

    def test_forced_change_point_at_max_runlength(
        self, simple_algorithm: MockOnlineAlgorithm, data_provider: MockDataProvider
    ) -> None:
        """Change point should be forced when run_length exceeds max_runlength."""
        detector = OnlineResetDetector(simple_algorithm, max_runlength=3)
        trace = detector.detect(data_provider)
        assert 3 in trace.forced_change_points

    def test_skip_period_suppresses_detections(
        self, counting_algorithm: MockOnlineAlgorithm, data_provider: MockDataProvider
    ) -> None:
        """Skip period should suppress detections after a change point."""
        detector = OnlineResetDetector(counting_algorithm, skip_period=2, threshold=0.5)
        counting_algorithm._config = MockAlgorithmConfiguration(return_value=1.0)
        trace = detector.detect(data_provider)
        assert trace.detected_change_points[0] == 0
        assert trace.skip_periods[0] == (1, 2)

    def test_skip_period_resets_after_reaching_threshold(self) -> None:
        algorithm = MockOnlineAlgorithm(MockAlgorithmConfiguration(return_value=1.0))
        detector = OnlineResetDetector(algorithm, skip_period=2, threshold=0.5)

        trace = detector.detect(MockDataProvider([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

        assert trace.detected_change_points == [0, 3]
        assert trace.skip_periods == [(1, 2), (4, 5)]
        assert trace.algorithm_states[1] is None
        assert trace.algorithm_states[2] is None
        assert trace.algorithm_states[3] is not None

    def test_collect_states_false_omits_state(
        self, simple_algorithm: MockOnlineAlgorithm, data_provider: MockDataProvider
    ) -> None:
        """When collect_states=False, algorithm_state should be None."""
        detector = OnlineResetDetector(simple_algorithm, collect_states=False)
        trace = detector.detect(data_provider)
        for state in trace.algorithm_states:
            assert state is None

    def test_collect_states_true_includes_state(
        self, simple_algorithm: MockOnlineAlgorithm, data_provider: MockDataProvider
    ) -> None:
        """When collect_states=True, algorithm_state should be set."""
        detector = OnlineResetDetector(simple_algorithm, collect_states=True)
        trace = detector.detect(data_provider)
        for state in trace.algorithm_states:
            assert state is not None

    def test_reset_after_detection(
        self, simple_algorithm: MockOnlineAlgorithm, data_provider: MockDataProvider
    ) -> None:
        """Algorithm should be reset after each change point."""
        detector = OnlineResetDetector(simple_algorithm, max_runlength=3)
        trace = detector.detect(data_provider)

        forced_idx = trace.forced_change_points[0]

        assert trace.algorithm_states[forced_idx - 1].call_count == forced_idx

        assert trace.algorithm_states[forced_idx] is not None
        assert trace.algorithm_states[forced_idx].call_count == forced_idx + 1
