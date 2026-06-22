# -*- coding: ascii -*-
"""
Tests for slope kl symbolic divergence contract.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.algorithms.online import SlopeKLSymbolicDivergence
from tests.contracts.core.test_online_algorithm_contract import OnlineAlgorithmContract


class TestSlopeKLSymbolicDivergenceContract(OnlineAlgorithmContract):
    @pytest.fixture
    def algorithm(self) -> SlopeKLSymbolicDivergence:
        return SlopeKLSymbolicDivergence(learning_period_size=4, delta=0.0, gamma=1.0)

    @pytest.fixture
    def sample_observation(self) -> float:
        return 1.0

    @pytest.fixture
    def update_observations(self) -> list[float]:
        return [0.0, 1.0, 2.0, 3.0, 10.0, -5.0]

    @pytest.fixture
    def fresh_state(self):
        return SlopeKLSymbolicDivergence(learning_period_size=4, delta=0.0, gamma=1.0).state
