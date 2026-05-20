# -*- coding: ascii -*-
"""
Gaussian maximum-likelihood estimating schema.

This module provides :class:`GaussianMLESchema`, which maintains online
estimates of mean and covariance using numerically stable incremental updates.
"""

from collections.abc import Sequence
from typing import cast

import numpy as np

from pysatl_cpd.algorithms.online.cusum.abstracts.estimator import IEstimatingSchema, ISchemaEstimates
from pysatl_cpd.algorithms.online.cusum.utils import coerce_observation
from pysatl_cpd.typedefs import MultivariateNumericArray, NumericArray, UnivariateNumericArray

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


class EstimatesGaussianMLE(ISchemaEstimates):
    mean: UnivariateNumericArray
    cov: MultivariateNumericArray


class GaussianMLESchema(IEstimatingSchema[UnivariateNumericArray, EstimatesGaussianMLE]):
    """
    Gaussian mean/covariance estimator.

    Parameters
    ----------
    adaptive
        Whether to update estimates online after training. Default is
        ``True``.
    """

    def __init__(self, adaptive: bool = True) -> None:
        self.adaptive = adaptive
        self._len = 0
        self._dim = -1
        self._welford_mean: NumericArray = np.zeros((0,), dtype=np.float64)
        self._welford_m2: NumericArray = np.zeros((0, 0), dtype=np.float64)

    def train(self, train_set: Sequence[UnivariateNumericArray]) -> None:
        """Initialise mean/covariance accumulators from a training sample.

        Parameters
        ----------
        train_set
            Training observations of shape ``(n_samples, dim)``.
        """
        self._dim = coerce_observation(train_set[0]).shape[0]
        self._len = len(train_set)

        _train_set = np.asarray([coerce_observation(obs) for obs in train_set])

        self._welford_mean = _train_set.mean(axis=0)

        x_centred = _train_set - self._welford_mean
        self._welford_m2 = x_centred.T @ x_centred

    def update(self, observation: UnivariateNumericArray) -> None:
        """Update running mean/covariance with one observation.

        Uses Welford's online algorithm. No-op when *adaptive* is ``False``.

        Parameters
        ----------
        observation
            New observation vector.
        """
        if not self.adaptive:
            return
        self._len += 1
        dx = observation - self._welford_mean
        self._welford_mean = self._welford_mean + dx / self._len
        dy = observation - self._welford_mean
        self._welford_m2 = self._welford_m2 + np.outer(dx, dy)

    @property
    def mean(self) -> UnivariateNumericArray:
        """Current mean estimate.

        Returns
        -------
        UnivariateNumericArray
            Mean vector.
        """
        return cast(UnivariateNumericArray, np.asarray(self._welford_mean))

    @property
    def cov(self) -> MultivariateNumericArray:
        """Current covariance estimate (unbiased).

        Returns
        -------
        MultivariateNumericArray
            Covariance matrix of shape ``(dim, dim)``.
        """
        return cast(MultivariateNumericArray, np.asarray(self._welford_m2 / (self._len - 1)))

    @property
    def estimates(self) -> EstimatesGaussianMLE:
        """Current estimated parameters as a typed dict.

        Returns
        -------
        EstimatesGaussianMLE
        """
        return {"mean": self.mean, "cov": self.cov}

    def reset(self) -> None:
        """Reset all accumulators to initial (untrained) state.

        Returns
        -------
        None
        """
        self._len = 0
        self._dim = -1
        self._welford_mean = np.zeros((0,), dtype=np.float64)
        self._welford_m2 = np.zeros((0, 0), dtype=np.float64)
