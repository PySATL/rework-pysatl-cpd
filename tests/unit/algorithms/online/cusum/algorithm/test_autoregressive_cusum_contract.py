# -*- coding: ascii -*-
"""
Tests for autoregressive cusum contract.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.algorithms.online import AutoregressiveCUSUM
from tests.contracts.core.test_online_algorithm_contract import OnlineAlgorithmContract


class TestAutoregressiveCUSUMContract(OnlineAlgorithmContract):
    @pytest.fixture
    def algorithm(self) -> AutoregressiveCUSUM:
        return AutoregressiveCUSUM(learning_period_size=5, delta=0.1, autoreg_order=1, adaptive_estimation=False)

    @pytest.fixture
    def sample_observation(self) -> float:
        return 1.0

    @pytest.fixture
    def update_observations(self) -> list[float]:
        return [1.0, 1.2, 1.4, 1.6, 1.8]

    @pytest.fixture
    def fresh_state(self):
        return AutoregressiveCUSUM(learning_period_size=5, delta=0.1, autoreg_order=1, adaptive_estimation=False).state
