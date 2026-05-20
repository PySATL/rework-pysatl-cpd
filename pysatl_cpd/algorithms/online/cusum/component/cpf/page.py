# -*- coding: ascii -*-
"""
Univariate Page CUSUM change-point function.

This module provides :class:`ChangepointFuncUnivariatePageCUSUM`, supporting
positive, negative, or two-sided Page CUSUM statistics.
"""

from typing import Literal

from pysatl_cpd.algorithms.online.cusum.abstracts.changepoint_func import ICusumChangepointFunc
from pysatl_cpd.typedefs import UnivariateNumericArray

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


class ChangepointFuncUnivariatePageCUSUM(ICusumChangepointFunc[UnivariateNumericArray]):
    """
    Univariate Page CUSUM statistic.

    Parameters
    ----------
    delta
        Reference drift/sensitivity parameter. Default is ``0.0``.
    side
        Detection side:

        - ``"pos"``: positive shifts only,
        - ``"neg"``: negative shifts only,
        - ``"both"``: two-sided detection.

        Default is ``"both"``.
    """

    def __init__(self, delta: float = 0.0, side: Literal["pos", "neg", "both"] = "both") -> None:
        self.delta = delta
        self.stat_pos = 0.0
        self.stat_neg = 0.0
        self.side = side

    def update(self, observation: UnivariateNumericArray) -> None:
        """Update one- or two-sided Page CUSUM statistics.

        Parameters
        ----------
        observation
            New univariate observation (must be dim=1).

        Raises
        ------
        ValueError
            If *observation* is not dim=1.
        """
        if observation.shape[0] != 1:
            raise ValueError("shape mismatch")  # TODO: add more context
        if self.side in {"pos", "both"}:
            self.stat_pos = max(0.0, self.stat_pos + float(observation.item()) - self.delta)
        if self.side in {"neg", "both"}:
            self.stat_neg = max(0.0, self.stat_neg - float(observation.item()) - self.delta)

    @property
    def value(self) -> float:
        """Current Page CUSUM statistic: max of positive and negative accumulators.

        Returns
        -------
        float
        """
        return max(self.stat_pos, self.stat_neg)

    def reset(self) -> None:
        """Reset positive and negative statistics to zero.

        Returns
        -------
        None
        """
        self.stat_pos = 0.0
        self.stat_neg = 0.0
