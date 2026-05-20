# -*- coding: ascii -*-
"""
Tests for gaussian conjugate.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.algorithms.online.bayesian.component.likelihood import GaussianConjugate


class TestGaussianConjugate:
    def test_predict_returns_prior_then_posterior_scores(self) -> None:
        likelihood = GaussianConjugate(mu_0=0.0, k_0=1.0, alpha_0=1.0, beta_0=1.0)

        before = likelihood.predict(np.float64(0.5))
        likelihood.update(np.float64(0.5))
        after = likelihood.predict(np.float64(0.5))

        assert before.shape == (1,)
        assert after.shape == (2,)

    def test_validation_raises_value_error_when_k0_zero(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            GaussianConjugate(mu_0=0.0, k_0=0.0, alpha_0=1.0, beta_0=1.0)
        assert "k_0" in str(exc_info.value)

    def test_validation_raises_value_error_when_alpha0_zero(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            GaussianConjugate(mu_0=0.0, k_0=1.0, alpha_0=0.0, beta_0=1.0)
        assert "alpha_0" in str(exc_info.value)

    def test_validation_raises_value_error_when_beta0_zero(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            GaussianConjugate(mu_0=0.0, k_0=1.0, alpha_0=1.0, beta_0=0.0)
        assert "beta_0" in str(exc_info.value)
