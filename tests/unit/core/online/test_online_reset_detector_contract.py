# -*- coding: ascii -*-
"""
Tests for online reset detector contract.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.core.online import OnlineResetDetector
from tests.contracts.core.test_change_point_detector_contract import ChangePointDetectorContract
from tests.contracts.core.test_online_detector_contract import OnlineDetectorContract
from tests.support.algorithms import CountingAlgorithm, CountingAlgorithmConfig


class TestOnlineResetDetectorContract(OnlineDetectorContract):
    @pytest.fixture
    def detector(self) -> OnlineResetDetector:
        return OnlineResetDetector(CountingAlgorithm(CountingAlgorithmConfig(threshold=5.0)), threshold=5.0)


class TestOnlineResetDetectorAsChangePointDetectorContract(ChangePointDetectorContract):
    @pytest.fixture
    def detector(self) -> OnlineResetDetector:
        return OnlineResetDetector(CountingAlgorithm(CountingAlgorithmConfig(threshold=4.0)), threshold=4.0)


def test_online_reset_detector_clone_keeps_detector_configuration() -> None:
    detector = OnlineResetDetector(
        CountingAlgorithm(CountingAlgorithmConfig(threshold=4.0)),
        threshold=4.0,
        skip_period=2,
        max_runlength=7,
        collect_states=False,
    )
    clone = detector.clone()
    assert clone.threshold == detector.threshold
    assert clone.skip_period == detector.skip_period
    assert clone.max_runlength == detector.max_runlength
    assert clone.collect_states == detector.collect_states
