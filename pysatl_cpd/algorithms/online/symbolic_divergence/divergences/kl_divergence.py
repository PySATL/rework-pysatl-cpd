# -*- coding: ascii -*-
"""
Kullback-Leibler divergence for the Symbolic Divergence algorithm.

This module provides :class:`KLDivergence`, a stateless implementation of the
Kullback-Leibler divergence ``D_KL(p || q) = sum(p * log(p / q))``. Inputs may
be raw symbol frequencies (counts) or probabilities; they are normalised
internally. Additive smoothing is applied to the reference distribution to keep
it strictly positive, and the divergence is evaluated in a numerically stable
log-difference form that tolerates zero counts and very large counts.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np

from pysatl_cpd.algorithms.online.symbolic_divergence.divergences.base import IDivergence
from pysatl_cpd.typedefs import UnivariateNumericArray


class KLDivergence(IDivergence):
    """Kullback-Leibler divergence with additive smoothing of the reference.

    Inputs may be raw symbol frequencies (counts) or probabilities; both are
    normalised internally. Only the reference distribution is smoothed by
    ``smoothing`` so that symbols unobserved during the learning period do not
    produce ``log(0)`` or division by zero. Symbols with zero empirical
    frequency contribute nothing (``0 * log 0 = 0``) and are skipped.

    The divergence is evaluated as

    .. math::
        D = \\frac{\\sum_i p_i (\\log p_i - \\log q_i)}{\\sum_i p_i}

    over the smoothed/normalised distributions, which stays numerically stable
    for frequencies that differ by many orders of magnitude.

    Parameters
    ----------
    smoothing
        Non-negative additive smoothing constant applied to the reference.

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
        """Compute ``D_KL(empirical || reference)``.

        Parameters
        ----------
        empirical
            Empirical symbol frequencies (raw counts or probabilities).
            Non-negative; need not sum to one.
        reference
            Reference symbol frequencies (raw counts or probabilities).
            Non-negative; need not sum to one.

        Returns
        -------
        float
        """
        p = np.asarray(empirical, dtype=np.float64)
        q = np.asarray(reference, dtype=np.float64)

        positive = p > 0.0
        if not np.any(positive):
            return 0.0

        p_total = float(p.sum())
        q_total = float(q.sum()) + self._smoothing * float(q.size)

        p_pos = p[positive]
        q_pos = q[positive] + self._smoothing

        log_p = np.log(p_pos) - np.log(p_total)
        log_q = np.log(q_pos) - np.log(q_total)
        return float(np.sum(p_pos * (log_p - log_q)) / p_total)

    def __repr__(self) -> str:
        return f"KLDivergence(smoothing={self._smoothing!r})"
