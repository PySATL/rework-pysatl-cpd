# -*- coding: ascii -*-
"""
Gaussian monitoring schema for generalized CUSUM.

This module provides :class:`GaussianMonitoringSchema`, which whitens
observation residuals using the inverse square root of covariance.
"""

from typing import cast

import numpy as np

from pysatl_cpd.algorithms.online.cusum.abstracts.monitoring import IMonitoringSchema
from pysatl_cpd.algorithms.online.cusum.component.estimator.gaussian_mle import EstimatesGaussianMLE
from pysatl_cpd.algorithms.online.cusum.utils import coerce_observation
from pysatl_cpd.typedefs import NumericArray, UnivariateNumericArray

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


class GaussianMonitoringSchema(IMonitoringSchema[UnivariateNumericArray, EstimatesGaussianMLE, UnivariateNumericArray]):
    """
    Gaussian monitoring transformation based on mean and covariance estimates.

    Parameters
    ----------
    cov_reg
        Diagonal covariance regularization added before matrix inversion.
    """

    def __init__(self, cov_reg: float) -> None:
        self.dim = -1
        self.cov_reg = cov_reg

    def evaluate(self, observation: UnivariateNumericArray, parameters: EstimatesGaussianMLE) -> UnivariateNumericArray:
        """Transform observation into whitened monitoring residual.

        Computes ``cov^{-1/2} @ (observation - mean)``.

        Parameters
        ----------
        observation
            New observation vector.
        parameters
            Estimates dict containing ``"mean"`` and ``"cov"``.

        Returns
        -------
        UnivariateNumericArray
            Whitened residual vector.
        """
        obs = coerce_observation(observation)
        if self.dim == -1:
            self.dim = obs.shape[0]

        mean = parameters["mean"]
        cov = parameters["cov"]
        return self._inv_mat_sqrt(cov) @ (obs - mean)

    def _inv_mat_sqrt(self, mat: NumericArray) -> UnivariateNumericArray:
        """Compute inverse square root of a regularised symmetric matrix.

        Parameters
        ----------
        mat
            Symmetric matrix to invert.

        Returns
        -------
        UnivariateNumericArray
        """
        _mat = 0.5 * (mat + mat.T) + (self.cov_reg) * np.eye(self.dim)
        W, V = np.linalg.eigh(_mat)
        W = np.clip(W, 1e-12, None)

        return cast(UnivariateNumericArray, V @ np.diag(W**-0.5) @ V.T)

    def reset(self) -> None:
        """Reset internal dimensionality tracker.

        Notes
        -----
        The monitoring transform itself is stateless; only the cached
        dimension is reset.

        Returns
        -------
        None
        """
        self.dim = -1
