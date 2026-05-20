# -*- coding: ascii -*-
"""
Tests for shewhart.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math

import pytest

from pysatl_cpd.algorithms.online.control_charts.shewhart_control_chart import (
    ShewhartControlChart,
    ShewhartControlChartConfiguration,
    ShewhartControlChartState,
)


class TestShewhartControlChartConfiguration:
    """Validation of ShewhartControlChartConfiguration."""

    def test_learning_period_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="learning_period_size"):
            ShewhartControlChartConfiguration(learning_period_size=0, window_size=3)

    def test_window_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            ShewhartControlChartConfiguration(learning_period_size=5, window_size=0)

    def test_window_size_cannot_exceed_learning_period(self) -> None:
        with pytest.raises(ValueError, match="window_size.*less than or equal"):
            ShewhartControlChartConfiguration(learning_period_size=3, window_size=5)

    def test_valid_configuration_is_created(self) -> None:
        config = ShewhartControlChartConfiguration(learning_period_size=10, window_size=5)
        assert config.learning_period_size == 10
        assert config.window_size == 5

    def test_hash_equal_for_identical_configs(self) -> None:
        c1 = ShewhartControlChartConfiguration(learning_period_size=10, window_size=5)
        c2 = ShewhartControlChartConfiguration(learning_period_size=10, window_size=5)
        assert hash(c1) == hash(c2)

    def test_hash_differs_for_different_configs(self) -> None:
        c1 = ShewhartControlChartConfiguration(learning_period_size=10, window_size=5)
        c2 = ShewhartControlChartConfiguration(learning_period_size=20, window_size=5)
        assert hash(c1) != hash(c2)

    def test_repr(self) -> None:
        config = ShewhartControlChartConfiguration(learning_period_size=10, window_size=5)
        assert repr(config) == "w = 5"


class TestShewhartControlChartState:
    """State snapshot tests."""

    def test_hash(self) -> None:
        state = ShewhartControlChartState(
            is_in_learning_period=True,
            mean=1.0,
            variance=0.5,
            samples_count=10,
            window_contents=[1.0, 2.0, 3.0],
        )
        # Should not raise
        h = hash(state)
        assert isinstance(h, int)

    def test_computed_properties(self) -> None:
        state = ShewhartControlChartState(
            mean=2.0,
            variance=4.0,
            samples_count=5,
            window_contents=[1.0, 2.0, 3.0],
        )
        assert state.standard_deviation == 2.0
        assert state.window_sum == 6.0
        assert state.window_size == 3
        assert state.window_mean == 2.0

    def test_window_mean_when_empty(self) -> None:
        state = ShewhartControlChartState(window_contents=[])
        assert state.window_mean == 0.0
        assert state.window_size == 0
        assert state.window_sum == 0.0

    def test_hash_uses_stable_hash(self) -> None:
        """State hash depends on attributes, not object identity."""
        s1 = ShewhartControlChartState(
            is_in_learning_period=True, mean=0.0, variance=0.0, samples_count=0, window_contents=[]
        )
        s2 = ShewhartControlChartState(
            is_in_learning_period=True, mean=0.0, variance=0.0, samples_count=0, window_contents=[]
        )
        assert hash(s1) == hash(s2)


class TestShewhartControlChartAlgorithm:
    """Core algorithm flow."""

    def test_returns_zero_during_learning_period(self) -> None:
        chart = ShewhartControlChart(learning_period_size=5, window_size=3)
        for i in range(5):
            result = chart.process(float(i))
            assert result == 0.0, f"Expected 0 during learning at step {i}, got {result}"

    def test_returns_finite_after_learning_period(self) -> None:
        chart = ShewhartControlChart(learning_period_size=5, window_size=3)
        for i in range(5):
            chart.process(float(i))
        # After learning period
        result = chart.process(5.0)
        assert math.isfinite(result)
        assert result > 0.0

    def test_detection_zero_when_std_dev_zero(self) -> None:
        """When all observations are identical, std_dev stays 0 and detection remains 0."""
        chart = ShewhartControlChart(learning_period_size=5, window_size=3)
        for _ in range(5):
            chart.process(1.0)
        # After learning period, std_dev is still 0 (all identical values)
        result = chart.process(1.0)
        assert result == 0.0
        # More identical observations
        for _ in range(10):
            result = chart.process(1.0)
            assert result == 0.0

    def test_detection_increases_with_larger_deviation(self) -> None:
        """Larger deviations produce larger statistics."""
        chart = ShewhartControlChart(learning_period_size=10, window_size=5)
        # Train on small values
        for _ in range(10):
            chart.process(1.0)

        # After learning, a large deviation
        small_dev = chart.process(1.5)
        big_dev = chart.process(10.0)
        assert math.isfinite(small_dev)
        assert math.isfinite(big_dev)
        assert big_dev > small_dev

    def test_state_reflects_internal_values(self) -> None:
        chart = ShewhartControlChart(learning_period_size=5, window_size=3)
        for i in range(7):
            chart.process(float(i))
        state = chart.state
        assert state.samples_count == 7
        assert state.is_in_learning_period is False  # 7 > 5


class TestShewhartControlChartEdgeCases:
    """Edge cases for process()."""

    def test_single_observation(self) -> None:
        """One observation processed."""
        chart = ShewhartControlChart(learning_period_size=5, window_size=3)
        result = chart.process(42.0)
        assert result == 0.0

    def test_window_size_equals_learning_period(self) -> None:
        """Boundary case where window_size == learning_period_size."""
        chart = ShewhartControlChart(learning_period_size=3, window_size=3)
        for i in range(3):
            assert chart.process(float(i)) == 0.0
        result = chart.process(10.0)
        assert math.isfinite(result)

    def test_large_number_of_observations(self) -> None:
        """Stability with many observations."""
        chart = ShewhartControlChart(learning_period_size=10, window_size=5)
        for i in range(100):
            result = chart.process(float(i))
            if i < 10:
                assert result == 0.0
            else:
                assert math.isfinite(result)

    def test_alternating_values(self) -> None:
        """Test with oscillating input."""
        chart = ShewhartControlChart(learning_period_size=5, window_size=3)
        for i in range(5):
            chart.process(float(i % 2))
        # After learning
        result = chart.process(0.0)
        assert math.isfinite(result)
