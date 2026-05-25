# -*- coding: ascii -*-
"""
Slope encoder for the Symbolic Divergence algorithm.

This module provides :class:`SlopeEncoder`, the canonical ``k = 2`` special
case of :class:`ISymbolEncoder`. It maps the difference between two
consecutive observations to one of five ordinal symbols
``{-2, -1, 0, 1, 2}`` using two thresholds.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence

from pysatl_cpd.algorithms.online.symbolic_divergence.encoders.base import ISymbolEncoder
from pysatl_cpd.typedefs import Number


class SlopeEncoder(ISymbolEncoder[int]):
    """Two-observation slope encoder ``h(x_i, x_{i+1}) = encode(x_{i+1} - x_i)``.

    The signed difference ``d = x_{i+1} - x_i`` is mapped to a symbol via:

    - ``d < -gamma``           -> ``-2``
    - ``-gamma <= d < -delta`` -> ``-1``
    - ``-delta <= d <= delta`` -> ``0``
    - ``delta < d <= gamma``   -> ``1``
    - ``d > gamma``            -> ``2``

    Parameters
    ----------
    delta
        Inner (dead-zone) threshold for the flat symbol. Must be non-negative.
    gamma
        Outer threshold for steep symbols. Must be strictly greater than
        ``delta``.

    Raises
    ------
    ValueError
        If ``delta`` is negative or ``gamma`` is not strictly greater than
        ``delta``.
    """

    _SYMBOLS: tuple[int, ...] = (-2, -1, 0, 1, 2)

    def __init__(self, delta: float, gamma: float) -> None:
        if delta < 0.0:
            raise ValueError(f"delta must be non-negative, got {delta}")
        if gamma <= delta:
            raise ValueError(f"gamma ({gamma}) must be strictly greater than delta ({delta})")

        self._delta = delta
        self._gamma = gamma

    @property
    def window_size(self) -> int:
        """Number of observations per window (``k = 2``).

        Returns
        -------
        int
        """
        return 2

    @property
    def num_symbols(self) -> int:
        """Size of the symbol alphabet (``|S| = 5``).

        Returns
        -------
        int
        """
        return len(self._SYMBOLS)

    @property
    def symbols(self) -> tuple[int, ...]:
        """Ordinal symbols in canonical order ``(-2, -1, 0, 1, 2)``.

        Returns
        -------
        tuple[int, ...]
        """
        return self._SYMBOLS

    def encode(self, window: Sequence[Number]) -> int:
        """Encode the slope of the last two observations into a symbol.

        Parameters
        ----------
        window
            Sequence whose last two elements are used as ``(x_i, x_{i+1})``.

        Returns
        -------
        int

        Raises
        ------
        ValueError
            If ``window`` contains fewer than two observations.
        """
        if len(window) < 2:
            raise ValueError(f"window must contain at least 2 observations, got {len(window)}")

        diff = window[-1] - window[-2]
        if diff < -self._gamma:
            return -2
        if diff < -self._delta:
            return -1
        if diff <= self._delta:
            return 0
        if diff <= self._gamma:
            return 1
        return 2

    def reset(self) -> None:
        """Reset internal state (the slope encoder is stateless).

        Returns
        -------
        None
        """
        return None

    def __repr__(self) -> str:
        return f"SlopeEncoder(delta={self._delta!r}, gamma={self._gamma!r})"
