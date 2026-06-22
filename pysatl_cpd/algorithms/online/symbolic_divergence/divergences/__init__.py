# -*- coding: ascii -*-
"""
Divergence functions for the Symbolic Divergence algorithm.

This package provides the divergence interface and concrete divergences that
quantify the discrepancy between an empirical symbol distribution and a
reference distribution.

.. raw:: html

    <h2>Public API</h2>

- ``IDivergence`` -- Protocol for divergences between discrete distributions.
  Any class implementing ``compute`` and ``reset`` is compatible.
- ``KLDivergence`` -- Kullback-Leibler divergence with additive smoothing.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.symbolic_divergence.divergences.base import IDivergence
from pysatl_cpd.algorithms.online.symbolic_divergence.divergences.kl_divergence import KLDivergence

__all__ = ["IDivergence", "KLDivergence"]
