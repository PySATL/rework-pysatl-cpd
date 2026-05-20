# -*- coding: ascii -*-
"""
Tests for detection trace.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.online.detectors.online_detection_trace import (
    OnlineDetectionStepResult,
    OnlineDetectionTrace,
    extract_periods,
)


@pytest.fixture
def sample_description() -> ChangePointDetectorDescription:
    """Provide a sample detector description."""
    return ChangePointDetectorDescription(name="TestDetector")


@pytest.fixture
def sample_steps() -> list[OnlineDetectionStepResult]:
    """Provide sample step results."""
    return [
        OnlineDetectionStepResult(step_num=0, detection_function=0.1),
        OnlineDetectionStepResult(step_num=1, detection_function=0.3),
        OnlineDetectionStepResult(step_num=2, detection_function=0.8, is_signal_change_point=True),
        OnlineDetectionStepResult(step_num=3, detection_function=0.2, is_in_skip_period=True),
        OnlineDetectionStepResult(step_num=4, detection_function=0.4),
    ]


class TestExtractPeriods:
    """Tests for extract_periods helper function."""

    def test_empty_sequence(self) -> None:
        """Empty sequence should return empty list."""
        assert extract_periods([]) == []

    def test_no_true_values(self) -> None:
        """No True values should return empty list."""
        assert extract_periods([False, False, None, False]) == []

    def test_single_true(self) -> None:
        """Single True should create a period of length 1."""
        periods = extract_periods([False, True, False])
        assert periods == [(1, 1)]

    def test_continuous_true(self) -> None:
        """Continuous True values should create a single period."""
        periods = extract_periods([False, True, True, True, False])
        assert periods == [(1, 3)]

    def test_multiple_periods(self) -> None:
        """Multiple True regions should create multiple periods."""
        periods = extract_periods([True, True, False, True, True, True])
        assert periods == [(0, 1), (3, 5)]

    def test_none_values_ignored(self) -> None:
        """None values should be ignored."""
        periods = extract_periods([True, None, True, False, True])
        assert periods == [(0, 0), (2, 2), (4, 4)]

    def test_all_true(self) -> None:
        """All True should create one period from 0 to end."""
        periods = extract_periods([True, True, True])
        assert periods == [(0, 2)]


class TestOnlineDetectionTraceFromRun:
    """Tests for OnlineDetectionTrace.from_run() class method."""

    def test_from_run_with_steps(
        self, sample_description: ChangePointDetectorDescription, sample_steps: list[OnlineDetectionStepResult]
    ) -> None:
        """from_run should correctly aggregate step results."""
        trace = OnlineDetectionTrace.from_run(steps=sample_steps, detector_description=sample_description)
        assert isinstance(trace, OnlineDetectionTrace)
        assert len(trace.detection_function) == 5
        assert len(trace.processing_time) == 5

    def test_from_run_detects_change_points(self, sample_description: ChangePointDetectorDescription) -> None:
        """Change points should be correctly identified."""
        steps = [
            OnlineDetectionStepResult(step_num=0, is_signal_change_point=True),
            OnlineDetectionStepResult(step_num=1),
            OnlineDetectionStepResult(step_num=2, is_forced_change_point=True),
        ]
        trace = OnlineDetectionTrace.from_run(steps=steps, detector_description=sample_description)
        assert trace.detected_change_points == [0, 2]
        assert trace.signal_change_points == [0]
        assert trace.forced_change_points == [2]

    def test_from_run_detection_function(
        self, sample_description: ChangePointDetectorDescription, sample_steps: list[OnlineDetectionStepResult]
    ) -> None:
        """detection_function should contain all step values."""
        trace = OnlineDetectionTrace.from_run(steps=sample_steps, detector_description=sample_description)
        expected = np.array([0.1, 0.3, 0.8, 0.2, 0.4])
        np.testing.assert_array_almost_equal(trace.detection_function, expected)

    def test_from_run_processing_time(
        self, sample_description: ChangePointDetectorDescription, sample_steps: list[OnlineDetectionStepResult]
    ) -> None:
        """processing_time should be array of correct length."""
        trace = OnlineDetectionTrace.from_run(steps=sample_steps, detector_description=sample_description)
        assert len(trace.processing_time) == 5

    def test_from_run_skip_periods(self, sample_description: ChangePointDetectorDescription) -> None:
        """Skip periods should be correctly extracted."""
        steps = [
            OnlineDetectionStepResult(step_num=0),
            OnlineDetectionStepResult(step_num=1, is_in_skip_period=True),
            OnlineDetectionStepResult(step_num=2, is_in_skip_period=True),
            OnlineDetectionStepResult(step_num=3),
        ]
        trace = OnlineDetectionTrace.from_run(steps=steps, detector_description=sample_description)
        assert trace.skip_periods == [(1, 2)]

    def test_from_run_stores_description(
        self, sample_description: ChangePointDetectorDescription, sample_steps: list[OnlineDetectionStepResult]
    ) -> None:
        """description should be stored."""
        trace = OnlineDetectionTrace.from_run(steps=sample_steps, detector_description=sample_description)
        assert trace.detector_description is sample_description


class TestOnlineDetectionTraceSlice:
    """Tests for OnlineDetectionTrace.cut() method."""

    def test_slice_adjusts_change_points(self, sample_description: ChangePointDetectorDescription) -> None:
        """Change points should be shifted relative to cut start."""
        trace = OnlineDetectionTrace(
            detector_description=sample_description,
            detected_change_points=[2, 5, 7],
            detection_function=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            processing_time=np.array([0.01] * 8),
            algorithm_states=[None] * 8,
        )
        sliced = trace.cut(2, 7)
        assert sliced.detected_change_points == [0, 3, 5]

    def test_slice_adjusts_skip_periods(self, sample_description: ChangePointDetectorDescription) -> None:
        """Skip periods should be shifted correctly."""
        trace = OnlineDetectionTrace(
            detector_description=sample_description,
            detected_change_points=[],
            detection_function=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            processing_time=np.array([0.01] * 5),
            algorithm_states=[None] * 5,
            skip_periods=[(1, 3)],
        )
        sliced = trace.cut(1, 4)
        assert sliced.skip_periods == [(0, 2)]

    def test_slice_out_of_range_periods_excluded(self, sample_description: ChangePointDetectorDescription) -> None:
        """Periods outside cut range should be excluded."""
        trace = OnlineDetectionTrace(
            detector_description=sample_description,
            detected_change_points=[],
            detection_function=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            processing_time=np.array([0.01] * 6),
            algorithm_states=[None] * 6,
            skip_periods=[(0, 1), (4, 5)],
        )
        sliced = trace.cut(2, 3)
        assert sliced.skip_periods == []

    def test_slice_data_function_sliced(self, sample_description: ChangePointDetectorDescription) -> None:
        """detection_function should be sliced correctly."""
        trace = OnlineDetectionTrace(
            detector_description=sample_description,
            detected_change_points=[],
            detection_function=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            processing_time=np.array([0.01] * 5),
            algorithm_states=[None] * 5,
        )
        sliced = trace.cut(1, 3)
        expected = np.array([0.2, 0.3, 0.4])
        np.testing.assert_array_almost_equal(sliced.detection_function, expected)

    def test_slice_processing_time_sliced(self, sample_description: ChangePointDetectorDescription) -> None:
        """processing_time should be sliced correctly."""
        trace = OnlineDetectionTrace(
            detector_description=sample_description,
            detected_change_points=[],
            detection_function=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            processing_time=np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
            algorithm_states=[None] * 5,
        )
        sliced = trace.cut(1, 3)
        expected = np.array([0.02, 0.03, 0.04])
        np.testing.assert_array_almost_equal(sliced.processing_time, expected)
