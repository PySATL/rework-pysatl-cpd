# -*- coding: ascii -*-
"""
Tests for variance cusum contract.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.algorithms.online import VarianceTwoSidedCUSUM
from tests.contracts.core.test_online_algorithm_contract import OnlineAlgorithmContract


class TestVarianceTwoSidedCUSUMContract(OnlineAlgorithmContract):
    @pytest.fixture
    def algorithm(self) -> VarianceTwoSidedCUSUM:
        return VarianceTwoSidedCUSUM(learning_period_size=4, delta=0.1, adaptive_estimation=False)

    @pytest.fixture
    def sample_observation(self) -> float:
        return 1.0

    @pytest.fixture
    def update_observations(self) -> list[float]:
        return [1.0, 1.1, 0.9, 1.0]

    @pytest.fixture
    def fresh_state(self):
        return VarianceTwoSidedCUSUM(learning_period_size=4, delta=0.1, adaptive_estimation=False).state
