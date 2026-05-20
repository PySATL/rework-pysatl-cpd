# -*- coding: ascii -*-
"""
Tests for change point detector contract.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.core.change_point_detector import ChangePointDetector, ChangePointDetectorDescription
from pysatl_cpd.core.detection_trace import DetectionTrace
from tests.support.providers import make_univariate_provider


class ChangePointDetectorContract:
    """Reusable checks for ChangePointDetector implementations."""

    @pytest.fixture
    def detector(self) -> ChangePointDetector:
        raise NotImplementedError

    @pytest.fixture
    def data_provider(self):
        return make_univariate_provider()

    def test_description_returns_description_object(self, detector: ChangePointDetector) -> None:
        assert isinstance(detector.description, ChangePointDetectorDescription)

    def test_description_name_non_empty(self, detector: ChangePointDetector) -> None:
        assert detector.description.name

    def test_detect_returns_detection_trace(self, detector: ChangePointDetector, data_provider) -> None:
        assert isinstance(detector.detect(data_provider), DetectionTrace)

    def test_trace_contains_detector_description(self, detector: ChangePointDetector, data_provider) -> None:
        trace = detector.detect(data_provider)
        assert trace.detector_description == detector.description

    def test_clone_returns_same_concrete_type(self, detector: ChangePointDetector) -> None:
        assert type(detector.clone()) is type(detector)

    def test_clone_is_independent_instance(self, detector: ChangePointDetector) -> None:
        assert detector.clone() is not detector
