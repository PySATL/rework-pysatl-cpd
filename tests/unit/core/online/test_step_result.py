# -*- coding: ascii -*-
"""
Tests for step result.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionStepResult


class TestOnlineDetectionStepResult:
    """Tests for OnlineDetectionStepResult."""

    def test_default_values(self) -> None:
        """Default values should be set correctly."""
        result = OnlineDetectionStepResult()
        assert result.step_num == 0
        assert not result.is_forced_change_point
        assert not result.is_signal_change_point
        assert not result.is_change_point
        assert not result.is_in_skip_period
        assert result.detection_function != result.detection_function  # nan check
        assert result.processing_time != result.processing_time  # nan check
        assert result.algorithm_state is None

    def test_is_change_point_true_for_forced(self) -> None:
        """is_change_point should be True if is_forced_change_point is True."""
        result = OnlineDetectionStepResult(is_forced_change_point=True)
        assert result.is_change_point

    def test_is_change_point_true_for_signal(self) -> None:
        """is_change_point should be True if is_signal_change_point is True."""
        result = OnlineDetectionStepResult(is_signal_change_point=True)
        assert result.is_change_point

    def test_is_change_point_true_for_both(self) -> None:
        """is_change_point should be True if both are True."""
        result = OnlineDetectionStepResult(is_forced_change_point=True, is_signal_change_point=True)
        assert result.is_change_point

    def test_is_change_point_false_for_neither(self) -> None:
        """is_change_point should be False if neither is True."""
        result = OnlineDetectionStepResult()
        assert not result.is_change_point

    def test_custom_step_num(self) -> None:
        """step_num should be settable."""
        result = OnlineDetectionStepResult(step_num=5)
        assert result.step_num == 5

    def test_custom_detection_function(self) -> None:
        """detection_function should be settable."""
        result = OnlineDetectionStepResult(detection_function=0.75)
        assert result.detection_function == 0.75

    def test_custom_processing_time(self) -> None:
        """processing_time should be settable."""
        result = OnlineDetectionStepResult(processing_time=0.001)
        assert result.processing_time == 0.001
