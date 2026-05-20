# -*- coding: ascii -*-
"""Two-sided variance CUSUM change-point detection algorithm."""

from dataclasses import dataclass
from typing import cast

import numpy as np

from pysatl_cpd.algorithms.online.cusum.abstracts.generalized_cusum import (
    GeneralizedCUSUM,
    GeneralizedCUSUMConfiguration,
    GeneralizedCUSUMState,
)
from pysatl_cpd.algorithms.online.cusum.component.cpf.page import ChangepointFuncUnivariatePageCUSUM
from pysatl_cpd.algorithms.online.cusum.component.estimator.gaussian_mle import EstimatesGaussianMLE, GaussianMLESchema
from pysatl_cpd.algorithms.online.cusum.component.monitoring.variance import VarianceMonitoringSchema
from pysatl_cpd.algorithms.online.cusum.utils import coerce_observation
from pysatl_cpd.typedefs import Number, UnivariateNumericArray, stable_hash

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


@dataclass(kw_only=True, frozen=True)
class VarianceTwoSidedCusumConfiguration(GeneralizedCUSUMConfiguration):
    """Configuration for the variance two-sided CUSUM algorithm.

    Attributes
    ----------
    delta
        Sensitivity parameter for the Page change-point function.
    """

    delta: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration after initialisation.

        Raises
        ------
        ValueError
            If *learning_period_size* is non-positive.
        """
        if self.learning_period_size <= 0:
            raise ValueError(f"learning_period_size must be positive, got {self.learning_period_size}")

    def __repr__(self) -> str:
        return f"delta={self.delta}"

    def __hash__(self) -> int:
        return stable_hash(
            (
                type(self).__module__,
                type(self).__qualname__,
                self.learning_period_size,
                self.adaptive_estimation,
                self.delta,
            )
        )


@dataclass(kw_only=True, frozen=True)
class VarianceTwoSidedCusumState(GeneralizedCUSUMState[EstimatesGaussianMLE]):
    """State snapshot of the variance two-sided CUSUM algorithm."""


class VarianceTwoSidedCUSUM(
    GeneralizedCUSUM[
        UnivariateNumericArray,
        VarianceTwoSidedCusumConfiguration,
        VarianceTwoSidedCusumState,
        EstimatesGaussianMLE,
        UnivariateNumericArray,
    ]
):
    """Two-sided CUSUM detector focused on variance changes.

    Parameters
    ----------
    learning_period_size
        Number of initial training observations (> 0).
    delta
        Sensitivity parameter for the Page CUSUM statistic.
    adaptive_estimation
        Whether to re-estimate variance online after training.
    """

    def __init__(
        self,
        learning_period_size: int,
        delta: float = 0.0,
        adaptive_estimation: bool = True,
    ) -> None:
        configuration = VarianceTwoSidedCusumConfiguration(
            learning_period_size=learning_period_size,
            delta=delta,
            adaptive_estimation=adaptive_estimation,
        )
        super().__init__(
            configuration=configuration,
            estimating_schema=GaussianMLESchema(adaptive=adaptive_estimation),
            monitoring_schema=VarianceMonitoringSchema(),
            changepoint_func=ChangepointFuncUnivariatePageCUSUM(delta=delta),
            adaptive_estimation=adaptive_estimation,
        )

    @property
    def name(self) -> str:
        """Human-readable algorithm name."""
        return "VarianceTwoSidedCUSUM"

    @property
    def configuration(self) -> VarianceTwoSidedCusumConfiguration:
        """Current algorithm configuration.

        Returns
        -------
        VarianceTwoSidedCusumConfiguration
        """
        return self._config

    @property
    def state(self) -> VarianceTwoSidedCusumState:
        """Materialise an immutable state snapshot.

        Returns
        -------
        VarianceTwoSidedCusumState
        """
        statistics = (
            self.estimates
            if len(self._train_X) >= self._config.learning_period_size
            else {"mean": np.array([], dtype=np.float64), "cov": np.zeros((0, 0), dtype=np.float64)}
        )
        return VarianceTwoSidedCusumState(is_in_learning_period=self._is_training, statistics=statistics)

    def process(self, observation: Number | UnivariateNumericArray) -> Number:
        """Ingest one observation and return the change-point statistic.

        Coerces input to a 1-D array and delegates to the parent.

        Parameters
        ----------
        observation
            New observation (must be dim=1).

        Returns
        -------
        Number

        Raises
        ------
        ValueError
            If *observation* is not dim=1.
        """
        obs = cast(UnivariateNumericArray, coerce_observation(observation))
        if obs.shape[0] != 1:
            raise ValueError(f"VarianceTwoSidedCUSUM only supports dim=1, got shape {obs.shape}")
        return super().process(obs)

    def __repr__(self) -> str:
        return f"VarianceTwoSidedCUSUM({self._config})"
