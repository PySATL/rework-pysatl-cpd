# -*- coding: ascii -*-
"""
Tests for univariate gaussian conjugate contract.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.algorithms.online import BayesianCPFType, UnivariateGaussianConjugateBOCPD
from tests.contracts.core.test_online_algorithm_contract import OnlineAlgorithmContract


class TestUnivariateGaussianConjugateBOCPDContract(OnlineAlgorithmContract):
    @pytest.fixture
    def algorithm(self) -> UnivariateGaussianConjugateBOCPD:
        return UnivariateGaussianConjugateBOCPD(
            learning_period_size=2,
            hazard_lambda=8.0,
            window=5,
            cpf_type=BayesianCPFType.MAX_RUN_LENGTH,
        )

    @pytest.fixture
    def sample_observation(self) -> float:
        return 0.1

    @pytest.fixture
    def fresh_state(self):
        return UnivariateGaussianConjugateBOCPD(
            learning_period_size=2,
            hazard_lambda=8.0,
            window=5,
            cpf_type=BayesianCPFType.MAX_RUN_LENGTH,
        ).state
