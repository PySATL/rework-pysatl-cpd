# -*- coding: ascii -*-
"""
Kullback-Leibler divergence for the Symbolic Divergence algorithm.

This module provides :class:`KLDivergence`, a stateless implementation of the
Kullback-Leibler divergence ``D_KL(p || q) = sum(p * log(p / q))`` with
additive smoothing to keep both distributions strictly positive.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np

from pysatl_cpd.algorithms.online.symbolic_divergence.divergences.base import IDivergence
from pysatl_cpd.typedefs import UnivariateNumericArray


class KLDivergence(IDivergence):
    """Kullback-Leibler divergence with additive smoothing.

    Both distributions are shifted by ``smoothing`` and renormalised before
    the divergence is evaluated, which avoids ``log(0)`` and division by zero
    when a symbol is unobserved in either distribution.

    Parameters
    ----------
    smoothing
        Non-negative additive smoothing constant. Must be non-negative.

    Raises
    ------
    ValueError
        If ``smoothing`` is negative.
    """

    def __init__(self, smoothing: float = 1e-10) -> None:
        if smoothing < 0.0:
            raise ValueError(f"smoothing must be non-negative, got {smoothing}")

        self._smoothing = smoothing

    def compute(self, empirical: UnivariateNumericArray, reference: UnivariateNumericArray) -> float:
        """Compute ``D_KL(empirical || reference)`` with smoothing.

        Parameters
        ----------
        empirical
            Empirical symbol frequencies. Non-negative and summing to one.
        reference
            Reference symbol probabilities. Non-negative and summing to one.

        Returns
        -------
        float
        """
        p = empirical + self._smoothing
        q = reference + self._smoothing
        p = p / p.sum()
        q = q / q.sum()
        return float(np.sum(p * np.log(p / q)))

    def reset(self) -> None:
        """Reset internal state (the divergence is stateless).

        Returns
        -------
        None
        """
        return None

    def __repr__(self) -> str:
        return f"KLDivergence(smoothing={self._smoothing!r})"
