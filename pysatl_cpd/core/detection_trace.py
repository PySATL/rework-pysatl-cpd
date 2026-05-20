# -*- coding: ascii -*-
"""
Module contains detection trace container for changepoint detection results.

This module provides a unified container for storing detection results from
both online and offline changepoint detection algorithms.
"""

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.typedefs import stable_hash


@dataclass(kw_only=True)
class DetectionTrace:
    """
    Container for changepoint detection algorithm output.

    This class stores the detected changepoint positions It serves
    as a unified result type for both online and offline detection
    algorithms.

    Attributes
    ----------
    detected_change_points
        Indices where changepoints were detected. For online algorithms,
        these are typically reported as they occur. For offline algorithms,
        these are the final changepoint positions.
    detector_description
        Description of the detector that produced the trace.

    Examples
    --------
    >>> trace = DetectionTrace(detected_change_points=[2, 4])
    >>> trace.detected_changes
    [2, 4]
    """

    detected_change_points: Sequence[int] = field(default_factory=list)
    detector_description: ChangePointDetectorDescription

    def __hash__(self) -> int:
        """Return a stable hash for the trace contents and detector identity."""
        return stable_hash(
            (
                type(self).__module__,
                type(self).__qualname__,
                self.detected_change_points,
                self.detector_description,
            )
        )

    def __post_init__(self) -> None:
        """
        Validate detection trace consistency.

        Ensures that:
        - Detected change-point indices are valid (non-negative)
        - All indices are unique

        Raises
        ------
        ValueError
            If any detected change index is negative, or if indices are
            not unique.
        """
        if not self.detected_change_points:
            return

        # Validate indices are non-negative
        if min(self.detected_change_points) < 0:
            raise ValueError(
                f"Detected change indices must be non-negative.Found negative index: {min(self.detected_change_points)}"
            )

        # Validate indices are unique
        if len(self.detected_change_points) != len(set(self.detected_change_points)):
            duplicates = [x for x in self.detected_change_points if list(self.detected_change_points).count(x) > 1]
            raise ValueError(f"Detected change indices must be unique. Found duplicates: {set(duplicates)}")

        # Check if indices are sorted
        if list(self.detected_change_points) != sorted(self.detected_change_points):
            warnings.warn(
                f"Detected change indices are not sorted: {list(self.detected_change_points)}. "
                "Consider sorting them for consistent behavior.",
                UserWarning,
                stacklevel=2,
            )
