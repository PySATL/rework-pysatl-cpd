# -*- coding: ascii -*-
"""
Tests for detection trace.
"""

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.typedefs import frozendict, stable_hash


@pytest.fixture
def sample_description() -> ChangePointDetectorDescription:
    """Provide a sample detector description."""
    return ChangePointDetectorDescription(name="TestDetector")


class TestDetectionTrace:
    """Tests for DetectionTrace validation and behavior."""

    def test_empty_trace(self, sample_description: ChangePointDetectorDescription) -> None:
        """Empty trace should not raise errors."""
        trace = DetectionTrace(
            detected_change_points=[],
            detector_description=sample_description,
        )
        assert trace.detected_change_points == []

    def test_valid_trace(self, sample_description: ChangePointDetectorDescription) -> None:
        """Valid trace with positive indices."""
        trace = DetectionTrace(
            detected_change_points=[2, 5, 7],
            detector_description=sample_description,
        )
        assert trace.detected_change_points == [2, 5, 7]

    def test_negative_index_raises_value_error(self, sample_description: ChangePointDetectorDescription) -> None:
        """Negative indices should raise ValueError."""
        with pytest.raises(ValueError, match="Detected change indices must be non-negative"):
            DetectionTrace(
                detected_change_points=[-1],
                detector_description=sample_description,
            )

    def test_duplicate_indices_raise_value_error(self, sample_description: ChangePointDetectorDescription) -> None:
        """Duplicate indices should raise ValueError."""
        with pytest.raises(ValueError, match="Detected change indices must be unique"):
            DetectionTrace(
                detected_change_points=[2, 2],
                detector_description=sample_description,
            )

    def test_unsorted_indices_warn(self, sample_description: ChangePointDetectorDescription) -> None:
        """Unsorted indices should raise a warning."""
        with pytest.warns(UserWarning, match="Detected change indices are not sorted"):
            DetectionTrace(
                detected_change_points=[5, 2, 7],
                detector_description=sample_description,
            )

    def test_detected_change_points_property(self, sample_description: ChangePointDetectorDescription) -> None:
        """detected_change_points should store the correct values."""
        trace = DetectionTrace(
            detected_change_points=[1, 3, 5],
            detector_description=sample_description,
        )
        assert trace.detected_change_points == [1, 3, 5]

    def test_detector_description_stored(self, sample_description: ChangePointDetectorDescription) -> None:
        """description should be stored correctly."""
        trace = DetectionTrace(
            detected_change_points=[],
            detector_description=sample_description,
        )
        assert trace.detector_description is sample_description

    def test_description_with_parameters(self) -> None:
        """Should handle non-empty frozen parameters."""
        description = ChangePointDetectorDescription(name="Test", parameters={"x": 1})
        assert description.parameters == {"x": 1}
        assert isinstance(description.parameters, frozendict)
        trace = DetectionTrace(
            detected_change_points=[],
            detector_description=description,
        )
        assert trace.detector_description.parameters == {"x": 1}

    def test_hash_uses_stable_hash(self, sample_description: ChangePointDetectorDescription) -> None:
        trace = DetectionTrace(
            detected_change_points=[1, 3, 5],
            detector_description=sample_description,
        )
        assert hash(trace) == stable_hash(
            (trace.__class__.__module__, trace.__class__.__qualname__, [1, 3, 5], sample_description)
        )
