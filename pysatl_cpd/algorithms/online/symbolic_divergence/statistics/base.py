# -*- coding: ascii -*-
"""
Protocol definition for change-point statistics.

This module defines :class:`IChangePointStatistic`, the structural typing
interface (PEP 544 protocol) for components that turn the stream of divergence
values into the final change-point statistic compared against a threshold by
the detector. It mirrors the role of the CUSUM change-point function.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Protocol


class IChangePointStatistic(Protocol):
    """Interface mapping a divergence stream into a change-point statistic."""

    def update(self, divergence: float, sample_size: int) -> None:
        """Incorporate a new divergence value into the statistic.

        Parameters
        ----------
        divergence
            Divergence between the empirical and reference distributions at the
            current step.
        sample_size
            Number of symbols accumulated so far (``n``), available for
            sample-size-dependent statistics such as ``2 n D``.
        """

    @property
    def value(self) -> float:
        """Current change-point statistic value.

        Returns
        -------
        float
        """

    def reset(self) -> None:
        """Reset internal state.

        Returns
        -------
        None
        """
