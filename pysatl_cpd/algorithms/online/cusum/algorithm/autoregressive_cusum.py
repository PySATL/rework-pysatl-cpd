# -*- coding: ascii -*-
"""Autoregressive CUSUM change-point detection algorithm."""

from dataclasses import dataclass
from typing import cast

import numpy as np

from pysatl_cpd.algorithms.online.cusum.abstracts.generalized_cusum import (
    GeneralizedCUSUM,
    GeneralizedCUSUMConfiguration,
    GeneralizedCUSUMState,
)
from pysatl_cpd.algorithms.online.cusum.component.cpf.page import ChangepointFuncUnivariatePageCUSUM
from pysatl_cpd.algorithms.online.cusum.component.estimator.gaussian_ar import EstimatesGaussianAR, GaussianARSchema
from pysatl_cpd.algorithms.online.cusum.component.monitoring.gaussian_arm import GaussianARMonitoringSchema
from pysatl_cpd.algorithms.online.cusum.utils import coerce_observation
from pysatl_cpd.typedefs import Number, UnivariateNumericArray, stable_hash

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


@dataclass(kw_only=True, frozen=True)
class AutoregressiveCusumConfiguration(GeneralizedCUSUMConfiguration):
    """Configuration for the autoregressive CUSUM algorithm.

    Attributes
    ----------
    delta
        Sensitivity parameter for the Page change-point function.
    autoreg_order
        Number of AR lags (> 0).
    autoreg_window
        Maximum observations retained for AR fitting (must exceed
        *autoreg_order* when provided).
    """

    delta: float = 0.0
    autoreg_order: int = 1
    autoreg_window: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialisation.

        Raises
        ------
        ValueError
            If *learning_period_size* is non-positive, *autoreg_order*
            is non-positive, *learning_period_size* is too small, or
            *autoreg_window* is too small.
        """
        if self.learning_period_size <= 0:
            raise ValueError(f"learning_period_size must be positive, got {self.learning_period_size}")
        if self.autoreg_order <= 0:
            raise ValueError(f"autoreg_order must be positive, got {self.autoreg_order}")
        min_learning_period_size = self.autoreg_order + 2
        if self.learning_period_size < min_learning_period_size:
            raise ValueError(
                "learning_period_size must be at least autoreg_order + 2, got "
                f"learning_period_size={self.learning_period_size}, autoreg_order={self.autoreg_order}"
            )
        if self.autoreg_window is not None and self.autoreg_window <= self.autoreg_order:
            raise ValueError(
                "autoreg_window must be greater than autoreg_order when provided, got "
                f"autoreg_window={self.autoreg_window}, autoreg_order={self.autoreg_order}"
            )

    def __repr__(self) -> str:
        return f"delta={self.delta}, order={self.autoreg_order}, window={self.autoreg_window}"

    def __hash__(self) -> int:
        return stable_hash(
            (
                type(self).__module__,
                type(self).__qualname__,
                self.learning_period_size,
                self.adaptive_estimation,
                self.delta,
                self.autoreg_order,
                self.autoreg_window,
            )
        )


@dataclass(kw_only=True, frozen=True)
class AutoregressiveCusumState(GeneralizedCUSUMState[EstimatesGaussianAR]):
    """State snapshot of the autoregressive CUSUM algorithm."""


class AutoregressiveCUSUM(
    GeneralizedCUSUM[
        UnivariateNumericArray,
        AutoregressiveCusumConfiguration,
        AutoregressiveCusumState,
        EstimatesGaussianAR,
        UnivariateNumericArray,
    ]
):
    """CUSUM detector for univariate autoregressive Gaussian time series.

    Parameters
    ----------
    learning_period_size
        Number of initial training observations.
    delta
        Sensitivity parameter for the Page CUSUM statistic.
    autoreg_order
        Number of AR lags (> 0).
    autoreg_window
        Max observations for AR fitting (``None`` = unbounded).
    adaptive_estimation
        Whether to re-fit AR coefficients online after training.
    """

    def __init__(
        self,
        learning_period_size: int,
        delta: float,
        autoreg_order: int = 1,
        autoreg_window: int | None = None,
        adaptive_estimation: bool = True,
    ) -> None:
        configuration = AutoregressiveCusumConfiguration(
            learning_period_size=learning_period_size,
            delta=delta,
            autoreg_order=autoreg_order,
            autoreg_window=autoreg_window,
            adaptive_estimation=adaptive_estimation,
        )
        super().__init__(
            configuration=configuration,
            estimating_schema=GaussianARSchema(
                autoreg_order=autoreg_order,
                adaptive=adaptive_estimation,
                window=autoreg_window,
            ),
            monitoring_schema=GaussianARMonitoringSchema(),
            changepoint_func=ChangepointFuncUnivariatePageCUSUM(delta=delta),
            adaptive_estimation=adaptive_estimation,
        )

    @property
    def name(self) -> str:
        """Human-readable algorithm name."""
        return "AutoregressiveCUSUM"

    @property
    def configuration(self) -> AutoregressiveCusumConfiguration:
        """Current algorithm configuration.

        Returns
        -------
        AutoregressiveCusumConfiguration
        """
        return self._config

    @property
    def state(self) -> AutoregressiveCusumState:
        """Materialise an immutable state snapshot.

        Returns
        -------
        AutoregressiveCusumState
        """
        statistics = (
            self.estimates
            if len(self._train_X) >= self._config.learning_period_size
            else {
                "intercept": 0.0,
                "coefficients": np.array([], dtype=np.float64),
                "noise_variance": 0.0,
                "history": np.array([], dtype=np.float64),
            }
        )
        return AutoregressiveCusumState(is_in_learning_period=self._is_training, statistics=statistics)

    def process(self, observation: Number | UnivariateNumericArray) -> Number:
        """Ingest one observation and return the change-point statistic.

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
            raise ValueError(f"AutoregressiveCUSUM only supports dim=1, got shape {obs.shape}")
        return super().process(obs)

    def __repr__(self) -> str:
        return f"AutoregressiveCUSUM({self._config})"
