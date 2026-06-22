# -*- coding: ascii -*-
"""
Protocol definition for divergence functions.

This module defines :class:`IDivergence`, the structural typing interface
(PEP 544 protocol) for divergences between two discrete probability
distributions over a symbol alphabet, consumed by the Symbolic Divergence
algorithm.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Protocol

from pysatl_cpd.typedefs import UnivariateNumericArray


class IDivergence(Protocol):
    """Interface for divergences between discrete symbol distributions.

    The two arguments are non-negative symbol frequencies: either raw counts
    or already-normalised probabilities. They need not sum to one;
    implementations are expected to normalise internally. The divergence is a
    pure function and therefore stateless (no ``reset`` is required).
    """

    def compute(self, empirical: UnivariateNumericArray, reference: UnivariateNumericArray) -> float:
        """Compute the divergence ``D(empirical || reference)``.

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
