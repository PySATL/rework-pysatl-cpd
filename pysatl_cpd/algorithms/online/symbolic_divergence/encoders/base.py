# -*- coding: ascii -*-
"""
Protocol definition for symbol encoders.

This module defines :class:`ISymbolEncoder`, the structural typing interface
(PEP 544 protocol) for window-to-symbol encodings ``h: R^k -> S`` consumed by
the Symbolic Divergence algorithm.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Hashable, Sequence
from typing import Protocol

from pysatl_cpd.typedefs import Number


class ISymbolEncoder[SymbolT: Hashable](Protocol):
    """Interface for window-to-symbol encodings ``h: R^k -> S``.

    Implementations map a fixed-size window of ``window_size`` consecutive
    observations to a single symbol drawn from a finite alphabet exposed via
    ``symbols``.
    """

    @property
    def window_size(self) -> int:
        """Number of observations per window (``k``).

        Returns
        -------
        int
        """

    @property
    def num_symbols(self) -> int:
        """Size of the symbol alphabet ``|S|``.

        Returns
        -------
        int
        """

    @property
    def symbols(self) -> tuple[SymbolT, ...]:
        """All possible symbols in canonical (index) order.

        Returns
        -------
        tuple[SymbolT, ...]
        """

    def encode(self, window: Sequence[Number]) -> SymbolT:
        """Encode a window of ``window_size`` observations into a symbol.

        Parameters
        ----------
        window
            Sequence of the most recent ``window_size`` observations.

        Returns
        -------
        SymbolT
        """

    def reset(self) -> None:
        """Reset internal encoder state, if any.

        Returns
        -------
        None
        """
