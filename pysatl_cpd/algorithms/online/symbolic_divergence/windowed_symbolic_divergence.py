# -*- coding: ascii -*-
"""
Windowed Symbolic Divergence online change-point detection algorithm.

This module provides :class:`WindowedSymbolicDivergence`, a generalized online
detector that encodes a sliding window of observations into a symbol, computes a
divergence between a recent-window empirical symbol distribution and a reference
distribution updated with symbols that leave that window, and maps that
divergence to a change-point statistic via a pluggable component.

It is a sibling of :class:`SymbolicDivergence`: the latter compares the running
(cumulative) empirical distribution against a reference fixed at the end of the
learning period, whereas this windowed variant compares a fixed-size recent
window against a reference that keeps growing with the observed past. The
encoding (``h: R^k -> S``), divergence, and change-point statistic remain
pluggable components.

:class:`WindowedSymbolicDivergence` is an abstract base that is generic over its
configuration and state types, mirroring the ``GeneralizedCUSUM`` design.
Concrete detectors (such as ``WindowedSlopeKLSymbolicDivergence``) subclass it,
bake their component parameters into a dedicated frozen configuration, and
implement the ``configuration`` and ``state`` properties. Keeping the component
parameters in the configuration is what lets distinct detector instances hash
differently in the benchmarking framework.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import abstractmethod
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
class WindowedSymbolicDivergenceState(OnlineAlgorithmState):
    """
    State snapshot of the windowed Symbolic Divergence algorithm.

    Attributes
    ----------
    samples_count
        Number of observations processed so far.
    symbol_counts
        Per-symbol counts in the current recent window, or the learning counts
        accumulated so far while still learning, in canonical (encoder) order.
    reference_distribution
        Current reference symbol distribution, or an empty tuple while still
        learning.
    """

    samples_count: int = 0
    symbol_counts: tuple[int, ...] = ()
    reference_distribution: tuple[float, ...] = ()

    def __hash__(self) -> int:
        """Return a stable hash for the windowed Symbolic Divergence state snapshot."""
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
class WindowedSymbolicDivergenceConfiguration(OnlineAlgorithmConfiguration):
    """
    Configuration parameters for the windowed Symbolic Divergence algorithm.

    The encoder, divergence, and statistic components are passed as objects to
    the algorithm and are intentionally not stored here, matching the component
    pattern used by the generalized CUSUM detectors.

    Attributes
    ----------
    recent_window_size
        Number of recent post-learning observations whose symbols form the
        empirical distribution compared against the reference.

    Raises
    ------
    ValueError
        If ``learning_period_size`` or ``recent_window_size`` is not positive.
    """

    recent_window_size: int = 0

    def __post_init__(self) -> None:
        """Validate configuration parameters.

        Raises
        ------
        ValueError
            If ``learning_period_size`` or ``recent_window_size`` is not
            positive.
        """
        if self.learning_period_size <= 0:
            raise ValueError(f"learning_period_size must be positive, got {self.learning_period_size}")
        if self.recent_window_size <= 0:
            raise ValueError(f"recent_window_size must be positive, got {self.recent_window_size}")

    def __repr__(self) -> str:
        """Return a short string representation of the configuration."""
        return f"learning_period_size={self.learning_period_size}, recent_window_size={self.recent_window_size}"

    def __hash__(self) -> int:
        """Return a stable hash for the windowed Symbolic Divergence configuration."""
        return stable_hash(
            (type(self).__module__, type(self).__qualname__, self.learning_period_size, self.recent_window_size)
        )


class WindowedSymbolicDivergence[
    ConfigurationT: WindowedSymbolicDivergenceConfiguration,
    StateT: WindowedSymbolicDivergenceState,
](OnlineAlgorithm[Number, ConfigurationT, StateT]):
    """
    Abstract windowed online change-point detector based on symbolic encoding.

    During the learning period the algorithm encodes each sliding window of
    ``encoder.window_size`` observations into a symbol and accumulates the
    symbol frequencies as the initial reference distribution. After the learning
    period it fills a fixed recent window of symbols, computes the divergence
    between that recent empirical distribution and the reference, and folds each
    symbol that drops out of the recent window into the reference, so the
    reference keeps growing with the observed past.

    The class is generic over its configuration and state types so that concrete
    detectors can extend both. Subclasses build a configuration that stores the
    component parameters and implement the ``configuration`` and ``state``
    properties. No threshold is applied here; thresholding is the responsibility
    of the detector (for example ``OnlineResetDetector``).

    Parameters
    ----------
    configuration
        Algorithm configuration. Its ``learning_period_size`` and
        ``recent_window_size`` must both be strictly greater than
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
        If ``configuration.learning_period_size`` or
        ``configuration.recent_window_size`` is not strictly greater than
        ``encoder.window_size``.
    """

    def __init__(
        self,
        configuration: ConfigurationT,
        encoder: ISymbolEncoder[int],
        divergence: IDivergence,
        statistic: IChangePointStatistic | None = None,
    ) -> None:
        if configuration.learning_period_size <= encoder.window_size:
            raise ValueError(
                f"learning_period_size ({configuration.learning_period_size}) must be strictly "
                f"greater than encoder.window_size ({encoder.window_size})"
            )
        if configuration.recent_window_size <= encoder.window_size:
            raise ValueError(
                f"recent_window_size ({configuration.recent_window_size}) must be strictly "
                f"greater than encoder.window_size ({encoder.window_size})"
            )

        self._configuration = configuration
        self._encoder = encoder
        self._divergence = divergence
        self._statistic: IChangePointStatistic = statistic if statistic is not None else RawDivergenceStatistic()
        self._window: deque[Number] = deque(maxlen=encoder.window_size)
        self._symbol_counts: list[int] = [0] * encoder.num_symbols
        self._recent_symbols_maxlen: int = configuration.recent_window_size - encoder.window_size + 1
        self._recent_symbols: deque[int] = deque(maxlen=self._recent_symbols_maxlen)
        self._reference_counts: UnivariateNumericArray | None = None
        self._reference_sample_size: int = 0
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
    def samples_count(self) -> int:
        """Number of observations processed so far.

        Returns
        -------
        int
        """
        return self._samples_count

    @property
    def symbol_counts(self) -> tuple[int, ...]:
        """Per-symbol counts in canonical (encoder) order.

        Returns
        -------
        tuple[int, ...]
        """
        return tuple(self._symbol_counts)

    @property
    def reference_distribution(self) -> tuple[float, ...]:
        """Current reference distribution.

        Returns
        -------
        tuple[float, ...]
        """
        return tuple(self._reference.tolist()) if self._reference is not None else ()

    @property
    def is_in_learning_period(self) -> bool:
        """Whether the algorithm is still within its learning period.

        Returns
        -------
        bool
        """
        return self._samples_count <= self._configuration.learning_period_size

    @property
    @abstractmethod
    def configuration(self) -> ConfigurationT: ...  # pragma: no cover

    @property
    @abstractmethod
    def state(self) -> StateT: ...  # pragma: no cover

    def process(self, observation: Number) -> Number:
        """Ingest one observation and return the change-point statistic.

        The sliding window is updated and, once full, encoded into a symbol.
        During learning, symbols build the initial reference. After learning,
        symbols populate a fixed recent window. Returns 0 until that window is
        full; afterwards computes the divergence between the recent distribution
        and the reference. When the recent window advances, the dropped symbol
        is folded into the reference.

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
        symbol_index = self._encoder.symbols.index(symbol)

        if self._samples_count == self._configuration.learning_period_size:
            self._symbol_counts[symbol_index] += 1
            learning_counts = np.asarray(self._symbol_counts, dtype=np.float64)
            self._reference_counts = learning_counts
            self._reference_sample_size = int(learning_counts.sum())
            self._reference = learning_counts / self._reference_sample_size
            self._symbol_counts = [0] * self._encoder.num_symbols
            self._window.clear()
            return 0.0

        if self._samples_count < self._configuration.learning_period_size:
            self._symbol_counts[symbol_index] += 1
            return 0.0

        reference_counts = self._reference_counts
        reference = self._reference
        if reference_counts is None or reference is None:
            return 0.0

        if len(self._recent_symbols) == self._recent_symbols_maxlen:
            dropped_symbol_index = self._recent_symbols.popleft()
            self._symbol_counts[dropped_symbol_index] -= 1
            reference_counts[dropped_symbol_index] += 1.0
            self._reference_sample_size += 1
            reference = reference_counts / self._reference_sample_size
            self._reference = reference

        self._recent_symbols.append(symbol_index)
        self._symbol_counts[symbol_index] += 1

        if len(self._recent_symbols) < self._recent_symbols_maxlen:
            return 0.0

        sample_size = len(self._recent_symbols)
        empirical: UnivariateNumericArray = np.asarray(self._symbol_counts, dtype=np.float64) / sample_size
        divergence = self._divergence.compute(empirical, reference)
        self._statistic.update(divergence, sample_size)

        return self._statistic.value

    def reset(self) -> None:
        """Reset the algorithm to its initial state.

        Clears the sliding window, recent window, symbol counts, reference
        distribution, and sample counter, and resets the encoder and statistic
        components.

        Returns
        -------
        None
        """
        self._encoder.reset()
        self._statistic.reset()
        self._window.clear()
        self._recent_symbols.clear()
        self._symbol_counts = [0] * self._encoder.num_symbols
        self._reference_counts = None
        self._reference_sample_size = 0
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
