# -*- coding: ascii -*-
"""
Tests for algorithm contract.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from tests.contracts.core.test_online_algorithm_contract import OnlineAlgorithmContract
from tests.support.algorithms import CountingAlgorithm, CountingAlgorithmConfig, CountingAlgorithmState


class TestCountingAlgorithmContract(OnlineAlgorithmContract):
    @pytest.fixture
    def algorithm(self) -> CountingAlgorithm:
        return CountingAlgorithm(CountingAlgorithmConfig(threshold=5.0))

    @pytest.fixture
    def sample_observation(self) -> float:
        return 2.0

    @pytest.fixture
    def fresh_state(self) -> CountingAlgorithmState:
        return CountingAlgorithmState()
