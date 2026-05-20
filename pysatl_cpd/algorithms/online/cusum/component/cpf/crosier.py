# -*- coding: ascii -*-
"""
Crosier CUSUM change-point function.

This module provides :class:`ChangepointFuncCrosierCUSUM`, which applies a
norm-based shrinkage update to the accumulated statistic.
"""

import numpy as np

from pysatl_cpd.algorithms.online.cusum.abstracts.changepoint_func import ICusumChangepointFunc
from pysatl_cpd.algorithms.online.cusum.utils import coerce_observation
from pysatl_cpd.typedefs import UnivariateNumericArray

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


class ChangepointFuncCrosierCUSUM(ICusumChangepointFunc[UnivariateNumericArray]):
    """
    Crosier-style CUSUM change-point statistic for vector observations.

    Parameters
    ----------
    dim
        Observation dimensionality.
    delta
        Shrinkage/sensitivity parameter controlling statistic contraction.
        Default is ``0.0``.
    """

    def __init__(self, dim: int, delta: float = 0.0) -> None:
        self.delta = delta

        self.dim = -1
        self.stat = np.zeros(
            0,
        )

    def update(self, observation: UnivariateNumericArray) -> None:
        """Update Crosier CUSUM statistic with a new observation.

        Applies norm-based shrinkage to the accumulated statistic.

        Parameters
        ----------
        observation
            New monitoring-space observation vector.
        """
        obs = coerce_observation(observation)
        if self.dim == -1:
            self.dim = obs.shape[0]
            self.stat = np.zeros(self.dim)

        stat_factor = max(1.0 - self.delta / float(np.linalg.norm(self.stat + observation)), 0.0)
        self.stat = stat_factor * (self.stat + observation)

    @property
    def value(self) -> float:
        """Current Crosier CUSUM statistic: Euclidean norm of the internal vector.

        Returns
        -------
        float
        """
        return float(np.linalg.norm(self.stat))

    def reset(self) -> None:
        """Reset internal accumulated statistic vector.

        Returns
        -------
        None
        """
        self.dim = -1
        self.stat = np.zeros(
            0,
        )
