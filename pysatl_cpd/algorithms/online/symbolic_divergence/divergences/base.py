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
    """Interface for divergences between discrete probability distributions."""

    def compute(self, empirical: UnivariateNumericArray, reference: UnivariateNumericArray) -> float:
        """Compute the divergence ``D(empirical || reference)``.

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

    def reset(self) -> None:
        """Reset internal state, if any.

        Returns
        -------
        None
        """
