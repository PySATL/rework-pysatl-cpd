# -*- coding: ascii -*-
"""
Tests for univariate gaussian conjugate.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.algorithms.online import UnivariateGaussianConjugateBOCPD
from pysatl_cpd.algorithms.online.bayesian.algorithm.abstract_bayesian import BayesianOnlineCPDState
from pysatl_cpd.algorithms.online.bayesian.algorithm.univariate_gaussian_conjugate import (
    UnivariateGaussianConjugateBOCPDConfiguration,
    UnivariateGaussianConjugateBOCPDState,
)


class TestUnivariateGaussianConjugateBOCPD:
    def test_process_returns_scalar_detection_values(self) -> None:
        algorithm = UnivariateGaussianConjugateBOCPD()

        values = [algorithm.process(value) for value in [0.1, 0.2, -0.3]]

        assert all(isinstance(value, np.floating | float) for value in values)
        assert algorithm.state.t == 3
        assert algorithm.state.mu_params.size == 3

    def test_learning_period_suppresses_initial_scores(self) -> None:
        algorithm = UnivariateGaussianConjugateBOCPD(learning_period_size=2)

        first = algorithm.process(0.1)
        second = algorithm.process(0.2)
        third = algorithm.process(0.3)

        assert first == pytest.approx(0.0)
        assert second == pytest.approx(0.0)
        assert float(third) >= 0.0

    def test_configuration_returns_correct_type(self) -> None:
        algorithm = UnivariateGaussianConjugateBOCPD()
        assert isinstance(algorithm.configuration, UnivariateGaussianConjugateBOCPDConfiguration)

    def test_state_returns_bayesian_state_type(self) -> None:
        algorithm = UnivariateGaussianConjugateBOCPD()
        algorithm.process(0.1)
        assert isinstance(algorithm.state, BayesianOnlineCPDState)

    def test_process_with_window_truncation_does_not_error(self) -> None:
        algorithm = UnivariateGaussianConjugateBOCPD(window=1)
        for _ in range(5):
            algorithm.process(0.1)


class TestUnivariateGaussianConjugateBOCPDConfiguration:
    def test_hazard_lambda_below_one_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            UnivariateGaussianConjugateBOCPDConfiguration(hazard_lambda=0.5)
        assert "hazard_lambda" in str(exc_info.value)

    def test_prior_k_zero_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            UnivariateGaussianConjugateBOCPDConfiguration(prior_k=0)
        assert "prior_k" in str(exc_info.value)

    def test_prior_alpha_zero_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            UnivariateGaussianConjugateBOCPDConfiguration(prior_alpha=0)
        assert "prior_alpha" in str(exc_info.value)

    def test_prior_beta_zero_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            UnivariateGaussianConjugateBOCPDConfiguration(prior_beta=0)
        assert "prior_beta" in str(exc_info.value)

    def test_hash_consistency(self) -> None:
        cfg1 = UnivariateGaussianConjugateBOCPDConfiguration()
        cfg2 = UnivariateGaussianConjugateBOCPDConfiguration()
        assert hash(cfg1) == hash(cfg2)


class TestUnivariateGaussianConjugateBOCPDState:
    def test_hash(self) -> None:
        state = UnivariateGaussianConjugateBOCPDState(
            t=3,
            run_length_log_posterior=[-0.1, -0.5],
            mu_params=np.array([0.0, 0.1]),
            k_params=np.array([1.0, 1.1]),
            alpha_params=np.array([1.0, 1.5]),
            beta_params=np.array([1.0, 1.2]),
        )
        _ = hash(state)
