# -*- coding: ascii -*-
"""
Sample-size-scaled change-point statistic for the Symbolic Divergence algorithm.

This module provides :class:`ScaledDivergenceStatistic`, which scales the
divergence by ``scale * sample_size``. With the default ``scale = 2.0`` and the
Kullback-Leibler divergence this yields the ``2 n D_n`` statistic studied in the
theory of the method.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.base import IChangePointStatistic


class ScaledDivergenceStatistic(IChangePointStatistic):
    """Statistic ``scale * sample_size * divergence``.

    Parameters
    ----------
    scale
        Non-negative multiplier. The default ``2.0`` reproduces the ``2 n D``
        statistic.

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
        """Scale the divergence by ``scale * sample_size``.

        Parameters
        ----------
        divergence
            Divergence at the current step.
        sample_size
            Number of accumulated symbols (``n``).
        """
        self._value = self._scale * sample_size * divergence

    @property
    def value(self) -> float:
        """Current scaled statistic value.

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
        return f"ScaledDivergenceStatistic(scale={self._scale!r})"
