# -*- coding: ascii -*-
"""Autoregressive Gaussian monitoring schema for generalized CUSUM."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np

from pysatl_cpd.algorithms.online.cusum.abstracts.monitoring import IMonitoringSchema
from pysatl_cpd.algorithms.online.cusum.component.estimator.gaussian_ar import EstimatesGaussianAR
from pysatl_cpd.algorithms.online.cusum.utils import coerce_observation
from pysatl_cpd.typedefs import UnivariateNumericArray


class GaussianARMonitoringSchema(
    IMonitoringSchema[UnivariateNumericArray, EstimatesGaussianAR, UnivariateNumericArray]
):
    """Standardised one-step autoregressive forecast residual.

    Computes ``(predicted - observed) / sqrt(noise_variance)`` using
    AR model coefficients and history from the estimating schema.
    """

    def evaluate(
        self,
        observation: UnivariateNumericArray,
        estimates: EstimatesGaussianAR,
    ) -> UnivariateNumericArray:
        """Compute standardised AR forecast residual.

        Parameters
        ----------
        observation
            New observation (must be dim=1).
        estimates
            AR estimates containing ``coefficients``, ``history``,
            ``intercept``, and ``noise_variance``.

        Returns
        -------
        UnivariateNumericArray
            Standardised residual array of length 1.

        Raises
        ------
        ValueError
            If *observation* is not dim=1, or if coefficients are
            not yet fitted.
        """
        obs = coerce_observation(observation)
        if obs.shape[0] != 1:
            raise ValueError(f"GaussianARMonitoringSchema only supports dim=1, got shape {obs.shape}")

        coefficients = np.asarray(estimates["coefficients"], dtype=np.float64)
        history = np.asarray(estimates["history"], dtype=np.float64)
        if coefficients.size == 0 or history.size != coefficients.size:
            raise ValueError("Autoregressive monitoring requires fitted coefficients and matching history")

        mean_val = float(estimates["intercept"] + np.dot(coefficients, history[::-1]))
        var_val = float(estimates["noise_variance"])
        residual = (mean_val - float(obs.item())) / np.sqrt(var_val + 1e-12)
        return np.array([residual], dtype=np.float64)

    def reset(self) -> None:
        """Reset monitoring schema state.

        Returns
        -------
        None
        """
        return None
