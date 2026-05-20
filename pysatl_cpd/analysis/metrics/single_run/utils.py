# -*- coding: ascii -*-

"""Shared matching helpers for single-run metrics."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence


def validate_error_margin(error_margin: tuple[int, int]) -> tuple[int, int]:
    """Validate that both margin components are non-negative.

    Parameters
    ----------
    error_margin
        Pair (left, right) margin bounds.

    Returns
    -------
    tuple[int, int]
        The validated margin tuple.

    Raises
    ------
    ValueError
        If either component is negative.
    """
    left, right = error_margin
    if left < 0 or right < 0:
        raise ValueError("The left and right margins must be non-negative numbers")
    return error_margin


def match_change_points(
    detected_change_points: Sequence[int],
    true_change_points: Sequence[int],
    error_margin: tuple[int, int],
) -> dict[int, set[int]]:
    """Match detections to true change points using greedy left-to-right assignment.

    For each true change point (in sorted order), the earliest unmatched
    detection within the error margin window is assigned.

    Parameters
    ----------
    detected_change_points
        Detected change-point indices.
    true_change_points
        Ground-truth change-point indices.
    error_margin
        (left, right) tolerance around each true change point.

    Returns
    -------
    dict[int, set[int]]
        Mapping from each true change point to the set of matched detections.
    """
    left, right = validate_error_margin(error_margin)

    sorted_true = sorted(true_change_points)
    sorted_detected = sorted(detected_change_points)
    used_detections: set[int] = set()
    matches_by_true: dict[int, set[int]] = {true_change: set() for true_change in sorted_true}

    for true_change in sorted_true:
        for detected_change in sorted_detected:
            if detected_change in used_detections:
                continue
            if true_change - left <= detected_change <= true_change + right:
                matches_by_true[true_change].add(detected_change)
                used_detections.add(detected_change)

    return {true_change: matches_by_true[true_change] for true_change in true_change_points}
