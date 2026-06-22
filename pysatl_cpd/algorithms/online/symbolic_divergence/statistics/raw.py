# -*- coding: ascii -*-
"""
Identity change-point statistic for the Symbolic Divergence algorithm.

This module provides :class:`RawDivergenceStatistic`, which returns the latest
divergence value unchanged. It is the default statistic and preserves the
algorithm's raw-divergence behaviour.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.base import IChangePointStatistic


class RawDivergenceStatistic(IChangePointStatistic):
    """Identity statistic: the value equals the latest divergence."""

    def __init__(self) -> None:
        self._value = 0.0

    def update(self, divergence: float, sample_size: int) -> None:
        """Store the latest divergence as the current statistic.

        Parameters
        ----------
        divergence
            Divergence at the current step.
        sample_size
            Number of accumulated symbols (unused by this statistic).
        """
        self._value = divergence

    @property
    def value(self) -> float:
        """Current statistic value (the latest divergence).

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
        return "RawDivergenceStatistic()"
