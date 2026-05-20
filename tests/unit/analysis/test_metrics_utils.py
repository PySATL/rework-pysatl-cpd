# -*- coding: ascii -*-
"""
Tests for metrics utils.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.utils import match
from pysatl_cpd.core.single_run import SingleRun
from tests.support.metrics import MockDetectionTrace, MockLabeledDataForMetrics


def test_module_imports() -> None:
    """Public API attributes are present."""
    import pysatl_cpd.analysis.metrics.utils as utils

    assert hasattr(utils, "__author__")
    assert hasattr(utils, "__copyright__")
    assert hasattr(utils, "__license__")
    assert callable(match)


def test_match_empty_detections() -> None:
    """No detections produces empty mapping."""
    run = SingleRun(
        trace=MockDetectionTrace(detected_change_points=[]),
        provider=MockLabeledDataForMetrics(change_points=[10, 50]),
    )
    result = match(run, error_margin=(5, 5))
    assert result == {10: set(), 50: set()}


def test_match_no_true_change_points() -> None:
    """No true change points produces empty mapping."""
    run = SingleRun(
        trace=MockDetectionTrace(detected_change_points=[10, 20]),
        provider=MockLabeledDataForMetrics(change_points=[]),
    )
    result = match(run, error_margin=(5, 5))
    assert result == {}


def test_match_perfect_matching() -> None:
    """Exact matches within margin."""
    run = SingleRun(
        trace=MockDetectionTrace(detected_change_points=[10, 50]),
        provider=MockLabeledDataForMetrics(change_points=[10, 50]),
    )
    result = match(run, error_margin=(5, 5))
    assert result == {10: {10}, 50: {50}}


def test_match_within_margin() -> None:
    """Detections within the error margin are matched."""
    run = SingleRun(
        trace=MockDetectionTrace(detected_change_points=[12, 48]),
        provider=MockLabeledDataForMetrics(change_points=[10, 50]),
    )
    result = match(run, error_margin=(5, 5))
    assert result == {10: {12}, 50: {48}}


def test_match_outside_margin() -> None:
    """Detections outside the error margin are not matched."""
    run = SingleRun(
        trace=MockDetectionTrace(detected_change_points=[0, 100]),
        provider=MockLabeledDataForMetrics(change_points=[10, 50]),
    )
    result = match(run, error_margin=(5, 5))
    assert result == {10: set(), 50: set()}


def test_match_greedy_left_to_right() -> None:
    """Greedy assignment: each true CP picks the earliest unmatched detection."""
    run = SingleRun(
        trace=MockDetectionTrace(detected_change_points=[10, 11]),
        provider=MockLabeledDataForMetrics(change_points=[10, 50]),
    )
    result = match(run, error_margin=(5, 5))
    assert result == {10: {10, 11}, 50: set()}


def test_match_asymmetric_margin() -> None:
    """Asymmetric error margin (left != right)."""
    run = SingleRun(
        trace=MockDetectionTrace(detected_change_points=[7, 15]),
        provider=MockLabeledDataForMetrics(change_points=[10]),
    )
    result = match(run, error_margin=(5, 2))
    assert result == {10: {7}}
