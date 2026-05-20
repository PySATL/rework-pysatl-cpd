# -*- coding: ascii -*-
"""
Tests for abstract bayesian.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.algorithms.online.bayesian.algorithm.abstract_bayesian import (
    AbstractBayesian,
    BayesianOnlineCPDConfiguration,
    BayesianOnlineCPDState,
)
from pysatl_cpd.algorithms.online.bayesian.component.cpf import MaxRunLengthCPF
from pysatl_cpd.algorithms.online.bayesian.component.hazard import ConstantHazard
from pysatl_cpd.algorithms.online.bayesian.component.likelihood import GaussianConjugate


class TestBayesianOnlineCPDConfiguration:
    def test_negative_learning_period_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            BayesianOnlineCPDConfiguration(learning_period_size=-1)
        assert "learning_period_size" in str(exc_info.value)

    def test_non_positive_window_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            BayesianOnlineCPDConfiguration(window=0)
        assert "window" in str(exc_info.value)

    def test_hash_consistency(self) -> None:
        cfg1 = BayesianOnlineCPDConfiguration()
        cfg2 = BayesianOnlineCPDConfiguration()
        assert hash(cfg1) == hash(cfg2)


class TestBayesianOnlineCPDState:
    def test_hash_with_posterior(self) -> None:
        state = BayesianOnlineCPDState(
            t=5,
            run_length_log_posterior=[-0.5, -1.0, -2.0],
        )
        _ = hash(state)


class TestAbstractBayesian:
    """Test base class paths not exercised by concrete implementations."""

    def test_base_recreate_returns_different_instance_with_t_zero(self) -> None:
        class _MinimalBayesian(AbstractBayesian):
            @property
            def name(self) -> str:
                return "_MinimalBayesian"

        config = BayesianOnlineCPDConfiguration()
        hazard = ConstantHazard(lambda_=10.0)
        likelihood = GaussianConjugate(mu_0=0.0, k_0=1.0, alpha_0=1.0, beta_0=1.0)
        cpf = MaxRunLengthCPF()
        algo = _MinimalBayesian(config, hazard, likelihood, cpf)
        assert algo.configuration is config
        algo.process(0.1)
        recreated = algo.recreate()
        assert recreated is not algo
        assert recreated.state.t == 0
