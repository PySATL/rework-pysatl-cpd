# -*- coding: ascii -*-
"""
Symbol encoders for the Symbolic Divergence algorithm.

This package provides the encoder interface and concrete encoders that map a
window of consecutive observations to a discrete symbol (``h: R^k -> S``).

.. raw:: html

    <h2>Public API</h2>

- ``ISymbolEncoder`` -- Protocol for window-to-symbol encodings. Any class
  implementing ``window_size``, ``num_symbols``, ``symbols``, ``encode``, and
  ``reset`` is compatible.
- ``SlopeEncoder`` -- Canonical ``k = 2`` encoder mapping the slope between two
  consecutive observations to ``{-2, -1, 0, 1, 2}``.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.symbolic_divergence.encoders.base import ISymbolEncoder
from pysatl_cpd.algorithms.online.symbolic_divergence.encoders.slope_encoder import SlopeEncoder

__all__ = ["ISymbolEncoder", "SlopeEncoder"]
