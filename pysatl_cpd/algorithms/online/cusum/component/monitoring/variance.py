# -*- coding: ascii -*-
"""
Variance monitoring schema for generalized CUSUM.

This module provides :class:`VarianceMonitoringSchema`, which converts
consecutive-observation differences into approximately standardized variance
monitoring statistics.
"""

from typing import cast

import numpy as np

from pysatl_cpd.algorithms.online.cusum.abstracts.monitoring import IMonitoringSchema
from pysatl_cpd.algorithms.online.cusum.component.estimator.gaussian_mle import EstimatesGaussianMLE
from pysatl_cpd.algorithms.online.cusum.utils import coerce_observation
from pysatl_cpd.typedefs import UnivariateNumericArray

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


class VarianceMonitoringSchema(IMonitoringSchema[UnivariateNumericArray, EstimatesGaussianMLE, UnivariateNumericArray]):
    """Variance-change monitoring transform for univariate observations."""

    def __init__(self) -> None:
        self._past_observation: UnivariateNumericArray | None = None

    def evaluate(
        self,
        observation: UnivariateNumericArray,
        estimates: EstimatesGaussianMLE,
    ) -> UnivariateNumericArray:
        """Compute variance monitoring residual for one observation.

        Uses consecutive-observation differences, standardised by
        the estimated variance, and applies a normalising transform.

        Parameters
        ----------
        observation
            New observation (must be dim=1).
        estimates
            Estimates dict containing ``"mean"`` and ``"cov"``.

        Returns
        -------
        UnivariateNumericArray
            Standardised variance-change residual.

        Raises
        ------
        ValueError
            If *observation* is not dim=1.
        """
        obs = cast(UnivariateNumericArray, coerce_observation(observation))
        if obs.shape[0] != 1:
            raise ValueError(f"VarianceMonitoringSchema only supports dim=1, got shape {obs.shape}")

        if self._past_observation is None:
            self._past_observation = cast(UnivariateNumericArray, np.asarray(estimates["mean"], dtype=np.float64))

        diff = np.abs((obs - self._past_observation) / np.sqrt(2.0))
        self._past_observation = obs.copy()

        cov = np.asarray(estimates["cov"], dtype=np.float64)
        cov_value = float(cov.reshape(-1)[0])
        residual = (np.sqrt(np.abs(diff / np.sqrt(cov_value))) - 0.82218) / 0.34914
        return cast(UnivariateNumericArray, np.asarray(residual, dtype=np.float64))

    def reset(self) -> None:
        """Clear the stored previous observation.

        Returns
        -------
        None
        """
        self._past_observation = None
