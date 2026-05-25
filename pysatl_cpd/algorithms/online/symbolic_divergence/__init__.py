# -*- coding: ascii -*-
"""
Symbolic Divergence online change-point detection.

This package implements :class:`SymbolicDivergence`, a generalized online
change-point detector that encodes a sliding window of observations into a
symbol via a pluggable encoder (``h: R^k -> S``) and monitors the divergence
between the empirical symbol distribution and a reference distribution learned
during the warm-up period.

The method is parameterized by two interchangeable components:

- a symbol encoder implementing ``ISymbolEncoder`` (the ``SlopeEncoder`` with
  ``k = 2`` is the canonical special case),
- a divergence function implementing ``IDivergence`` (Kullback-Leibler being
  one option).

Public exports are wired up once the algorithm and its components are in place.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"
