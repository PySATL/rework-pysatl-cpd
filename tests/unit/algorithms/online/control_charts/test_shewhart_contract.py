# -*- coding: ascii -*-
"""
Tests for shewhart contract.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.algorithms.online import ShewhartControlChart
from tests.contracts.core.test_online_algorithm_contract import OnlineAlgorithmContract


class TestShewhartControlChartContract(OnlineAlgorithmContract):
    @pytest.fixture
    def algorithm(self) -> ShewhartControlChart:
        return ShewhartControlChart(learning_period_size=3, window_size=2)

    @pytest.fixture
    def sample_observation(self) -> float:
        return 1.0

    @pytest.fixture
    def fresh_state(self):
        return ShewhartControlChart(learning_period_size=3, window_size=2).state
