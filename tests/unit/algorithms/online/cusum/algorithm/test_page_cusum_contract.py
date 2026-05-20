# -*- coding: ascii -*-
"""
Tests for page cusum contract.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.algorithms.online import PageTwoSidedCusum
from tests.contracts.core.test_online_algorithm_contract import OnlineAlgorithmContract


class TestPageTwoSidedCusumContract(OnlineAlgorithmContract):
    @pytest.fixture
    def algorithm(self) -> PageTwoSidedCusum:
        return PageTwoSidedCusum(learning_period_size=3, delta=0.1, adaptive_estimation=False)

    @pytest.fixture
    def sample_observation(self) -> np.ndarray:
        return np.array([1.0])

    @pytest.fixture
    def update_observations(self) -> list[np.ndarray]:
        return [np.array([1.0]), np.array([1.1]), np.array([0.9])]

    @pytest.fixture
    def fresh_state(self):
        return PageTwoSidedCusum(learning_period_size=3, delta=0.1, adaptive_estimation=False).state
