# -*- coding: ascii -*-
"""Univariate Gaussian conjugate Bayesian online change-point detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self, cast

import numpy as np

from pysatl_cpd.algorithms.online.bayesian._enum import BayesianCPFType
from pysatl_cpd.algorithms.online.bayesian.algorithm.abstract_bayesian import (
    AbstractBayesian,
    BayesianOnlineCPDConfiguration,
    BayesianOnlineCPDState,
)
from pysatl_cpd.algorithms.online.bayesian.component.cpf import DropCPF, MaxRunLengthCPF
from pysatl_cpd.algorithms.online.bayesian.component.hazard import ConstantHazard
from pysatl_cpd.algorithms.online.bayesian.component.likelihood import GaussianConjugate
from pysatl_cpd.typedefs import stable_hash

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


@dataclass(kw_only=True, frozen=True)
class UnivariateGaussianConjugateBOCPDConfiguration(BayesianOnlineCPDConfiguration):
    """Configuration for the univariate Gaussian conjugate BOCPD algorithm."""

    hazard_lambda: float = 10.0
    prior_mu: float = 0.0
    prior_k: float = 1.0
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration fields after initialisation.

        Extends parent validation with hazard and prior checks.

        Raises
        ------
        ValueError
            If *hazard_lambda* < 1.0 or any prior scale parameter is
            non-positive.
        """
        super().__post_init__()
        if self.hazard_lambda < 1.0:
            raise ValueError("hazard_lambda must be >= 1.0")
        if self.prior_k <= 0:
            raise ValueError("prior_k must be positive")
        if self.prior_alpha <= 0:
            raise ValueError("prior_alpha must be positive")
        if self.prior_beta <= 0:
            raise ValueError("prior_beta must be positive")

    def __hash__(self) -> int:
        return stable_hash(
            (
                type(self).__module__,
                type(self).__qualname__,
                self.learning_period_size,
                self.window,
                self.cpf_type,
                self.hazard_lambda,
                self.prior_mu,
                self.prior_k,
                self.prior_alpha,
                self.prior_beta,
            )
        )


@dataclass(kw_only=True, frozen=True)
class UnivariateGaussianConjugateBOCPDState(BayesianOnlineCPDState):
    """State for the univariate Gaussian conjugate BOCPD algorithm.

    Attributes
    ----------
    mu_params
        Posterior mean parameters per run length.
    k_params
        Posterior pseudo-count parameters per run length.
    alpha_params
        Posterior shape parameters per run length.
    beta_params
        Posterior scale parameters per run length.
    """

    mu_params: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    k_params: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    alpha_params: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    beta_params: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    def __hash__(self) -> int:
        return stable_hash(
            (
                type(self).__module__,
                type(self).__qualname__,
                self.is_in_learning_period,
                self.t,
                tuple(np.asarray(self.run_length_log_posterior, dtype=np.float64).tolist()),
                tuple(self.mu_params.tolist()),
                tuple(self.k_params.tolist()),
                tuple(self.alpha_params.tolist()),
                tuple(self.beta_params.tolist()),
            )
        )


class UnivariateGaussianConjugateBOCPD(AbstractBayesian):
    """Univariate Gaussian conjugate Bayesian online change-point detector.

    Wires a :class:`ConstantHazard`, :class:`GaussianConjugate` likelihood,
    and the selected CPF strategy.

    Parameters
    ----------
    learning_period_size
        Number of initial steps where the score is clamped to 0.
    hazard_lambda
        Expected run length for the constant hazard model (>= 1).
    prior_mu
        Prior mean for the Normal-Inverse-Gamma conjugate prior.
    prior_k
        Prior pseudo-count (> 0).
    prior_alpha
        Prior shape (> 0).
    prior_beta
        Prior scale (> 0).
    window
        Maximum number of run-length states to retain.
    cpf_type
        Change-point function variant (max-run-length or drop).
    """

    def __init__(
        self,
        learning_period_size: int = 0,
        hazard_lambda: float = 10.0,
        prior_mu: float = 0.0,
        prior_k: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        window: int | None = None,
        cpf_type: BayesianCPFType = BayesianCPFType.MAX_RUN_LENGTH,
    ) -> None:
        configuration = UnivariateGaussianConjugateBOCPDConfiguration(
            learning_period_size=learning_period_size,
            hazard_lambda=hazard_lambda,
            prior_mu=prior_mu,
            prior_k=prior_k,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            window=window,
            cpf_type=cpf_type,
        )

        cpf = MaxRunLengthCPF() if cpf_type == BayesianCPFType.MAX_RUN_LENGTH else DropCPF()
        super().__init__(
            configuration=configuration,
            hazard=ConstantHazard(lambda_=hazard_lambda),
            likelihood=GaussianConjugate(mu_0=prior_mu, k_0=prior_k, alpha_0=prior_alpha, beta_0=prior_beta),
            cpf=cpf,
        )
        self._config = configuration
        self._likelihood = cast(GaussianConjugate, self._likelihood)

    @property
    def name(self) -> str:
        return "UnivariateGaussianConjugateBOCPD"

    @property
    def configuration(self) -> UnivariateGaussianConjugateBOCPDConfiguration:
        """Return the concrete configuration type.

        Returns
        -------
        UnivariateGaussianConjugateBOCPDConfiguration
        """
        return cast(UnivariateGaussianConjugateBOCPDConfiguration, self._config)

    @property
    def state(self) -> UnivariateGaussianConjugateBOCPDState:
        """Materialise an immutable snapshot including likelihood parameters.

        Returns
        -------
        UnivariateGaussianConjugateBOCPDState
        """
        likelihood = cast(GaussianConjugate, self._likelihood)
        configuration = self.configuration
        return UnivariateGaussianConjugateBOCPDState(
            is_in_learning_period=self.t < configuration.learning_period_size,
            t=self.t,
            run_length_log_posterior=self._run_length_log_posterior.copy(),
            mu_params=likelihood._mu_params.copy(),
            k_params=likelihood._k_params.copy(),
            alpha_params=likelihood._alpha_params.copy(),
            beta_params=likelihood._beta_params.copy(),
        )

    def recreate(self) -> Self:
        """Create a fresh algorithm instance with identical configuration.

        Returns
        -------
        Self
        """
        configuration = self.configuration
        return type(self)(
            learning_period_size=configuration.learning_period_size,
            hazard_lambda=configuration.hazard_lambda,
            prior_mu=configuration.prior_mu,
            prior_k=configuration.prior_k,
            prior_alpha=configuration.prior_alpha,
            prior_beta=configuration.prior_beta,
            window=configuration.window,
            cpf_type=configuration.cpf_type,
        )

    def __repr__(self) -> str:
        return f"UnivariateGaussianConjugateBOCPD({self._config})"
