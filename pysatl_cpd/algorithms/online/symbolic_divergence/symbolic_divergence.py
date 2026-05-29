# -*- coding: ascii -*-
"""
Symbolic Divergence online change-point detection algorithm.

This module provides :class:`SymbolicDivergence`, a generalized online detector
that encodes a sliding window of observations into a symbol, computes a
divergence between the running empirical symbol distribution and a reference
distribution fixed at the end of the learning period, and maps that divergence
to a change-point statistic via a pluggable component.

The encoding (``h: R^k -> S``), divergence, and change-point statistic are
pluggable components, which makes the Slope-encoding + Kullback-Leibler +
``2 n D`` combination a single special case of a broader family.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Self

import numpy as np

from pysatl_cpd.algorithms.online.symbolic_divergence.divergences.base import IDivergence
from pysatl_cpd.algorithms.online.symbolic_divergence.encoders.base import ISymbolEncoder
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.base import IChangePointStatistic
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.raw import RawDivergenceStatistic
from pysatl_cpd.core.online.ionline_algorithm import (
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
)
from pysatl_cpd.typedefs import Number, UnivariateNumericArray, stable_hash


@dataclass(kw_only=True, frozen=True)
class SymbolicDivergenceState(OnlineAlgorithmState):
    """
    State snapshot of the Symbolic Divergence algorithm.

    Attributes
    ----------
    samples_count
        Number of observations processed so far.
    symbol_counts
        Cumulative per-symbol counts in canonical (encoder) order.
    reference_distribution
        Reference symbol distribution fixed at the end of the learning period,
        or an empty tuple while still learning.
    """

    samples_count: int = 0
    symbol_counts: tuple[int, ...] = ()
    reference_distribution: tuple[float, ...] = ()

    def __hash__(self) -> int:
        """Return a stable hash for the Symbolic Divergence state snapshot."""
        return stable_hash(
            (
                type(self).__module__,
                type(self).__qualname__,
                self.is_in_learning_period,
                self.samples_count,
                self.symbol_counts,
                self.reference_distribution,
            )
        )


@dataclass(kw_only=True, frozen=True)
class SymbolicDivergenceConfiguration(OnlineAlgorithmConfiguration):
    """
    Configuration parameters for the Symbolic Divergence algorithm.

    The encoder, divergence, and statistic components are passed as objects to
    the algorithm and are intentionally not stored here, matching the component
    pattern used by the generalized CUSUM detectors.

    Raises
    ------
    ValueError
        If ``learning_period_size`` is not positive.
    """

    def __post_init__(self) -> None:
        """Validate configuration parameters.

        Raises
        ------
        ValueError
            If ``learning_period_size`` is not positive.
        """
        if self.learning_period_size <= 0:
            raise ValueError(f"learning_period_size must be positive, got {self.learning_period_size}")

    def __repr__(self) -> str:
        """Return a short string representation of the configuration."""
        return f"learning_period_size={self.learning_period_size}"

    def __hash__(self) -> int:
        """Return a stable hash for the Symbolic Divergence configuration."""
        return stable_hash((type(self).__module__, type(self).__qualname__, self.learning_period_size))


class SymbolicDivergence(OnlineAlgorithm[Number, SymbolicDivergenceConfiguration, SymbolicDivergenceState]):
    """
    Online change-point detector based on symbolic encoding and divergence.

    During the learning period the algorithm encodes each sliding window of
    ``encoder.window_size`` observations into a symbol, accumulates the symbol
    frequencies, and fixes them as the reference distribution. After the
    learning period it computes the divergence between the running empirical
    distribution and that reference, then delegates the final emitted value to
    the change-point statistic component.

    No threshold is applied here; thresholding is the responsibility of the
    detector (for example ``OnlineResetDetector``).

    Parameters
    ----------
    learning_period_size
        Number of initial observations used to estimate the reference
        distribution. Must be positive and strictly greater than
        ``encoder.window_size``.
    encoder
        Window-to-symbol encoder implementing ``ISymbolEncoder``.
    divergence
        Divergence function implementing ``IDivergence``.
    statistic
        Change-point statistic implementing ``IChangePointStatistic`` that maps
        the divergence stream to the emitted value. Defaults to
        ``RawDivergenceStatistic`` (the raw divergence).

    Raises
    ------
    ValueError
        If ``learning_period_size`` is not positive, or is not strictly
        greater than ``encoder.window_size``.
    """

    def __init__(
        self,
        learning_period_size: int,
        encoder: ISymbolEncoder[int],
        divergence: IDivergence,
        statistic: IChangePointStatistic | None = None,
    ) -> None:
        self._configuration = SymbolicDivergenceConfiguration(learning_period_size=learning_period_size)
        if learning_period_size <= encoder.window_size:
            raise ValueError(
                f"learning_period_size ({learning_period_size}) must be strictly greater "
                f"than encoder.window_size ({encoder.window_size})"
            )

        self._encoder = encoder
        self._divergence = divergence
        self._statistic: IChangePointStatistic = statistic if statistic is not None else RawDivergenceStatistic()
        self._window: deque[Number] = deque(maxlen=encoder.window_size)
        self._symbol_counts: list[int] = [0] * encoder.num_symbols
        self._reference: UnivariateNumericArray | None = None
        self._samples_count: int = 0

    @property
    def encoder(self) -> ISymbolEncoder[int]:
        """Symbol encoder component.

        Returns
        -------
        ISymbolEncoder[int]
        """
        return self._encoder

    @property
    def divergence(self) -> IDivergence:
        """Divergence component.

        Returns
        -------
        IDivergence
        """
        return self._divergence

    @property
    def statistic(self) -> IChangePointStatistic:
        """Change-point statistic component.

        Returns
        -------
        IChangePointStatistic
        """
        return self._statistic

    @property
    def name(self) -> str:
        """Human-readable algorithm name.

        Returns
        -------
        str
        """
        return "SymbolicDivergence"

    @property
    def configuration(self) -> SymbolicDivergenceConfiguration:
        """Current algorithm configuration.

        Returns
        -------
        SymbolicDivergenceConfiguration
        """
        return self._configuration

    @property
    def state(self) -> SymbolicDivergenceState:
        """Materialise an immutable state snapshot.

        Returns
        -------
        SymbolicDivergenceState
        """
        return SymbolicDivergenceState(
            is_in_learning_period=self._samples_count <= self._configuration.learning_period_size,
            samples_count=self._samples_count,
            symbol_counts=tuple(self._symbol_counts),
            reference_distribution=(tuple(self._reference.tolist()) if self._reference is not None else ()),
        )

    def process(self, observation: Number) -> Number:
        """Ingest one observation and return the change-point statistic.

        The sliding window is updated and, once full, encoded into a symbol
        whose count is accumulated. Returns 0 during the learning period or
        before the first full window; afterwards computes the divergence
        between the running empirical distribution and the fixed reference and
        maps it to the emitted statistic.

        Parameters
        ----------
        observation
            New scalar observation.

        Returns
        -------
        Number
        """
        self._samples_count += 1
        self._window.append(observation)

        if len(self._window) < self._encoder.window_size:
            return 0.0

        symbol = self._encoder.encode(list(self._window))
        self._symbol_counts[self._encoder.symbols.index(symbol)] += 1
        sample_size = sum(self._symbol_counts)

        counts = np.asarray(self._symbol_counts, dtype=np.float64)
        empirical: UnivariateNumericArray = counts / sample_size

        if self._samples_count == self._configuration.learning_period_size:
            self._reference = empirical.copy()

        if self._samples_count <= self._configuration.learning_period_size or self._reference is None:
            return 0.0

        divergence = self._divergence.compute(empirical, self._reference)
        self._statistic.update(divergence, sample_size)
        return self._statistic.value

    def reset(self) -> None:
        """Reset the algorithm to its initial state.

        Clears the sliding window, symbol counts, reference distribution, and
        sample counter, and resets the encoder, divergence, and statistic
        components.

        Returns
        -------
        None
        """
        self._encoder.reset()
        self._divergence.reset()
        self._statistic.reset()
        self._window.clear()
        self._symbol_counts = [0] * self._encoder.num_symbols
        self._reference = None
        self._samples_count = 0

    def recreate(self) -> Self:
        """Create a fresh copy with identical configuration and reset state.

        Returns
        -------
        Self
        """
        algorithm = deepcopy(self)
        algorithm.reset()
        return algorithm

    def __repr__(self) -> str:
        """Return a string representation of the algorithm with its configuration."""
        return f"SymbolicDivergence({self._configuration})"
