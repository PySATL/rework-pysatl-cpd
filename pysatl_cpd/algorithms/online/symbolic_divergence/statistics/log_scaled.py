# -*- coding: ascii -*-
"""
Log-scaled change-point statistic for the Symbolic Divergence algorithm.

This module provides :class:`LogScaledDivergenceStatistic`, which scales the
divergence by ``scale * sample_size / log(sample_size + 1)``. Compared with the
linear ``scale * sample_size`` growth of :class:`ScaledDivergenceStatistic`, the
logarithmic denominator damps the statistic when the number of symbols backing
the empirical distribution is large, which is useful for the windowed detector.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math

from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.base import IChangePointStatistic


class LogScaledDivergenceStatistic(IChangePointStatistic):
    """Statistic ``scale * sample_size / log(sample_size + 1) * divergence``.

    Parameters
    ----------
    scale
        Non-negative multiplier. The default is ``2.0``.

    Raises
    ------
    ValueError
        If ``scale`` is negative.
    """

    def __init__(self, scale: float = 2.0) -> None:
        if scale < 0.0:
            raise ValueError(f"scale must be non-negative, got {scale}")

        self._scale = scale
        self._value = 0.0

    def update(self, divergence: float, sample_size: int) -> None:
        """Scale the divergence by ``scale * sample_size / log(sample_size + 1)``.

        Parameters
        ----------
        divergence
            Divergence at the current step.
        sample_size
            Number of symbols represented by the current empirical
            distribution. Must be positive.
        """
        self._value = self._scale * (sample_size / math.log(sample_size + 1)) * divergence

    @property
    def value(self) -> float:
        """Current log-scaled statistic value.

        Returns
        -------
        float
        """
        return self._value

    def reset(self) -> None:
        """Reset the stored value to zero.

        Returns
        -------
        None
        """
        self._value = 0.0

    def __repr__(self) -> str:
        return f"LogScaledDivergenceStatistic(scale={self._scale!r})"
