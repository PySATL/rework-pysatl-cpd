# -*- coding: ascii -*-
"""
Tests for conftest.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence

import pytest

from pysatl_cpd.core.single_run import SingleRun
from tests.support.metrics import MockDetectionTrace, MockLabeledDataForMetrics, MockOnlineDetectionTrace


@pytest.fixture
def mock_trace():
    """Factory for MockDetectionTrace."""

    def _make(detected: Sequence[int]) -> MockDetectionTrace:
        return MockDetectionTrace(detected_change_points=detected)

    return _make


@pytest.fixture
def mock_online_trace():
    """Factory for MockOnlineDetectionTrace."""

    def _make(detected: Sequence[int]) -> MockOnlineDetectionTrace:
        return MockOnlineDetectionTrace(detected_change_points=detected)

    return _make


@pytest.fixture
def mock_provider():
    """Factory for MockLabeledDataForMetrics."""

    def _make(change_points: Sequence[int], length: int = 100) -> MockLabeledDataForMetrics:
        return MockLabeledDataForMetrics(change_points=change_points, length=length)

    return _make


@pytest.fixture
def make_classification_run(mock_trace, mock_provider):
    """Factory for SingleRun with MockDetectionTrace."""

    def _make(
        detected: Sequence[int], true_cps: Sequence[int], length: int = 100
    ) -> SingleRun[MockDetectionTrace, MockLabeledDataForMetrics]:
        return SingleRun(
            trace=mock_trace(detected),
            provider=mock_provider(true_cps, length),
        )

    return _make


@pytest.fixture
def make_online_run(mock_online_trace, mock_provider):
    """Factory for SingleRun with MockOnlineDetectionTrace."""

    def _make(
        detected: Sequence[int], true_cps: Sequence[int], length: int = 100
    ) -> SingleRun[MockOnlineDetectionTrace, MockLabeledDataForMetrics]:
        return SingleRun(
            trace=mock_online_trace(detected),
            provider=mock_provider(true_cps, length),
        )

    return _make


@pytest.fixture
def make_multiple_runs(mock_trace, mock_provider):
    """Factory for multiple SingleRun objects (classification)."""

    def _make(
        specs: Sequence[tuple[Sequence[int], Sequence[int]]], length: int = 100
    ) -> list[SingleRun[MockDetectionTrace, MockLabeledDataForMetrics]]:
        return [
            SingleRun(
                trace=mock_trace(detected),
                provider=mock_provider(true_cps, length),
            )
            for detected, true_cps in specs
        ]

    return _make


@pytest.fixture
def make_online_multiple_runs(mock_online_trace, mock_provider):
    """Factory for multiple SingleRun objects (online)."""

    def _make(
        specs: Sequence[tuple[Sequence[int], Sequence[int]]], length: int = 100
    ) -> list[SingleRun[MockOnlineDetectionTrace, MockLabeledDataForMetrics]]:
        return [
            SingleRun(
                trace=mock_online_trace(detected),
                provider=mock_provider(true_cps, length),
            )
            for detected, true_cps in specs
        ]

    return _make
