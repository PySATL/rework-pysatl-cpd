# -*- coding: ascii -*-
"""
Tests for online detector.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC

import pytest

from pysatl_cpd.core.change_point_detector import ChangePointDetector
from pysatl_cpd.core.online import OnlineDetector
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace
from tests.support.core import ConcreteOnlineDetector, MockDataProvider, SimpleAlgorithm, SimpleAlgorithmConfig


class TestOnlineDetector:
    """Tests for OnlineDetector."""

    def test_is_abstract_base_class(self) -> None:
        assert issubclass(OnlineDetector, ABC)

    def test_contract_members_are_abstract(self) -> None:
        assert getattr(OnlineDetector.clone, "__isabstractmethod__", False)
        assert getattr(OnlineDetector._detect, "__isabstractmethod__", False)

    def test_cannot_instantiate_without_required_members(self) -> None:
        algorithm = SimpleAlgorithm(SimpleAlgorithmConfig())

        with pytest.raises(TypeError):
            OnlineDetector(algorithm)

    def test_concrete_detector_stores_algorithm(self) -> None:
        algorithm = SimpleAlgorithm(SimpleAlgorithmConfig())
        detector = ConcreteOnlineDetector(algorithm)
        assert detector.algorithm is algorithm

    def test_inherits_from_change_point_detector(self) -> None:
        """OnlineDetector should inherit from ChangePointDetector."""
        assert issubclass(OnlineDetector, ChangePointDetector)

    def test_concrete_detector_returns_online_trace(self) -> None:
        """Concrete implementation should return OnlineDetectionTrace."""
        algorithm = SimpleAlgorithm(SimpleAlgorithmConfig())
        detector = ConcreteOnlineDetector(algorithm)
        provider = MockDataProvider()
        result = detector.detect(provider)

        assert isinstance(result, OnlineDetectionTrace)

    def test_concrete_detector_clone_recreates_algorithm(self) -> None:
        algorithm = SimpleAlgorithm(SimpleAlgorithmConfig())
        detector = ConcreteOnlineDetector(algorithm)

        clone = detector.clone()

        assert clone is not detector
        assert clone.algorithm is not algorithm
