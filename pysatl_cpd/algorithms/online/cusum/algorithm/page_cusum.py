# -*- coding: ascii -*-
"""
Page two-sided CUSUM change-point detection algorithm.

This module provides :class:`PageTwoSidedCusum`, an online detector
based on the two-sided Page CUSUM statistic for Gaussian observations.
"""

from dataclasses import dataclass

from pysatl_cpd.algorithms.online.cusum.abstracts.generalized_cusum import (
    GeneralizedCUSUM,
    GeneralizedCUSUMConfiguration,
    GeneralizedCUSUMState,
)
from pysatl_cpd.algorithms.online.cusum.component.cpf.page import ChangepointFuncUnivariatePageCUSUM
from pysatl_cpd.algorithms.online.cusum.component.estimator.gaussian_mle import EstimatesGaussianMLE, GaussianMLESchema
from pysatl_cpd.algorithms.online.cusum.component.monitoring.gaussian import GaussianMonitoringSchema
from pysatl_cpd.typedefs import MultivariateNumericArray, UnivariateNumericArray

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


@dataclass(kw_only=True, frozen=True)
class PageTwoSidedCusumConfiguration(GeneralizedCUSUMConfiguration):
    """
    Configuration parameters for the Page two-sided CUSUM algorithm.

    Attributes
    ----------
    delta
        Sensitivity/reference parameter for the Page change-point function.
    cov_reg
        Covariance regularization coefficient used in monitoring.
    adaptive_estimation
        Whether Gaussian parameter estimation is adaptive.
    """

    delta: float = 0.0
    cov_reg: float = 1e-6
    adaptive_estimation: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters.

        Raises
        ------
        ValueError
            If *learning_period_size* is non-positive or *cov_reg* is
            non-positive.
        """
        if self.learning_period_size <= 0:
            raise ValueError(f"learning_period_size must be positive, got {self.learning_period_size}")

        if self.cov_reg <= 0:
            raise ValueError(f"cov_reg must be positive, got {self.cov_reg}")

    def __repr__(self) -> str:
        """Return a short string representation of the configuration."""
        return f"delta={self.delta}"


@dataclass(kw_only=True, frozen=True)
class PageTwoSidedCusumState(GeneralizedCUSUMState[EstimatesGaussianMLE]):
    """
    State snapshot of the Page two-sided CUSUM algorithm.
    """


class PageTwoSidedCusum(
    GeneralizedCUSUM[
        MultivariateNumericArray,
        PageTwoSidedCusumConfiguration,
        PageTwoSidedCusumState,
        EstimatesGaussianMLE,
        UnivariateNumericArray,
    ]
):
    """
    Two-sided Page CUSUM detector for Gaussian observations.

    This algorithm maintains running estimates of mean and covariance,
    computes whitened residuals, and tracks a two-sided Page CUSUM statistic.

    Parameters
    ----------
    learning_period_size
        Number of initial observations used for parameter learning.
    delta
        Sensitivity/reference parameter. Default is ``0.0``.
    cov_reg
        Covariance regularization. Default is ``1e-6``.
    adaptive_estimation
        Whether to update estimates online. Default is ``True``.
    """

    def __init__(
        self,
        learning_period_size: int,
        delta: float = 0.0,
        cov_reg: float = 1e-6,
        adaptive_estimation: bool = True,
    ) -> None:
        configuration = PageTwoSidedCusumConfiguration(
            learning_period_size=learning_period_size,
            delta=delta,
            cov_reg=cov_reg,
            adaptive_estimation=adaptive_estimation,
        )
        super().__init__(
            configuration=configuration,
            estimating_schema=GaussianMLESchema(adaptive=adaptive_estimation),  # type: ignore
            monitoring_schema=GaussianMonitoringSchema(cov_reg),  # type: ignore
            changepoint_func=ChangepointFuncUnivariatePageCUSUM(delta=delta),
        )

    @property
    def name(self) -> str:
        """Return the algorithm name."""
        return "PageTwoSidedCusum"

    @property
    def configuration(self) -> PageTwoSidedCusumConfiguration:
        """Return the algorithm configuration."""
        return self._config

    @property
    def state(self) -> PageTwoSidedCusumState:
        """Return the algorithm configuration."""
        return PageTwoSidedCusumState(is_in_learning_period=self._is_training, statistics=self.estimates)

    def __repr__(self) -> str:
        """Return a string representation of the algorithm with its configuration."""
        return f"PageTwoSidedCusum({self._config})"
