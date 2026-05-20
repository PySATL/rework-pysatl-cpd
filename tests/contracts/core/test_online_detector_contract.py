# -*- coding: ascii -*-
"""
Tests for online detector contract.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.core.online import OnlineDetectionTrace, OnlineDetector
from tests.support.providers import make_univariate_provider


class OnlineDetectorContract:
    """Reusable checks for OnlineDetector implementations."""

    @pytest.fixture
    def detector(self) -> OnlineDetector:
        raise NotImplementedError

    @pytest.fixture
    def data_provider(self):
        return make_univariate_provider((1.0, 2.0, 5.0, 6.0))

    def test_algorithm_reference_is_stored(self, detector: OnlineDetector) -> None:
        assert detector.algorithm is not None

    def test_detect_returns_online_detection_trace(self, detector: OnlineDetector, data_provider) -> None:
        assert isinstance(detector.detect(data_provider), OnlineDetectionTrace)

    def test_clone_recreates_independent_algorithm_instance(self, detector: OnlineDetector) -> None:
        clone = detector.clone()
        assert clone is not detector
        assert clone.algorithm is not detector.algorithm

    def test_clone_preserves_algorithm_configuration(self, detector: OnlineDetector) -> None:
        assert detector.clone().algorithm.configuration == detector.algorithm.configuration
