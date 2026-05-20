# -*- coding: ascii -*-
"""
Tests for metrics.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence


class MockDetectionTrace:
    """Trace exposing only detected change points."""

    def __init__(self, detected_change_points: Sequence[int]) -> None:
        self.detected_change_points = detected_change_points


class MockOnlineDetectionTrace:
    """Online trace exposing only detected change points."""

    def __init__(self, detected_change_points: Sequence[int]) -> None:
        self.detected_change_points = detected_change_points


class MockLabeledDataForMetrics:
    """Provider exposing only change points and length."""

    def __init__(self, change_points: Sequence[int], length: int = 100) -> None:
        self.change_points = change_points
        self._length = length

    def __len__(self) -> int:
        return self._length
