# -*- coding: ascii -*-
"""
Tests for ranges.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import (
    AutoThresholdsRange,
    LinearThresholdsRange,
    ManualThresholdsRange,
)


class TestManualThresholdsRange:
    def test_empty_values(self) -> None:
        rng = ManualThresholdsRange(_values=())
        assert rng.thresholds_range == ()

    def test_single_value(self) -> None:
        rng = ManualThresholdsRange(_values=(42.0,))
        assert rng.thresholds_range == (42.0,)

    def test_multiple_values(self) -> None:
        rng = ManualThresholdsRange(_values=(1.0, 2.5, 3.0))
        assert rng.thresholds_range == (1.0, 2.5, 3.0)


class TestLinearThresholdsRange:
    def test_default_values(self) -> None:
        rng = LinearThresholdsRange()
        assert len(rng.thresholds_range) == 10
        assert rng.thresholds_range[0] == 0.0
        assert rng.thresholds_range[-1] == 1.0

    def test_custom_values(self) -> None:
        rng = LinearThresholdsRange(start=0.5, end=2.0, count=4)
        assert rng.thresholds_range == (0.5, 1.0, 1.5, 2.0)

    def test_count_less_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="count must be >= 1"):
            LinearThresholdsRange(count=0)

    def test_count_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="count must be >= 1"):
            LinearThresholdsRange(count=-1)

    def test_count_one_returns_just_start(self) -> None:
        rng = LinearThresholdsRange(start=0.5, end=0.9, count=1)
        assert len(rng.thresholds_range) == 1
        assert rng.thresholds_range[0] == 0.5

    def test_start_equals_end(self) -> None:
        rng = LinearThresholdsRange(start=0.5, end=0.5, count=3)
        assert all(v == 0.5 for v in rng.thresholds_range)

    def test_start_greater_than_end_descending(self) -> None:
        rng = LinearThresholdsRange(start=1.0, end=0.0, count=5)
        assert rng.thresholds_range[0] == 1.0
        assert rng.thresholds_range[-1] == 0.0
        assert rng.thresholds_range[0] > rng.thresholds_range[-1]


class TestAutoThresholdsRange:
    def test_default_count(self) -> None:
        rng = AutoThresholdsRange()
        assert rng.count == 2
        assert rng.thresholds_range == ()

    def test_custom_count(self) -> None:
        rng = AutoThresholdsRange(count=5)
        assert rng.count == 5
        assert rng.thresholds_range == ()

    def test_count_less_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="count must be >= 1"):
            AutoThresholdsRange(count=0)

    def test_count_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="count must be >= 1"):
            AutoThresholdsRange(count=-1)
