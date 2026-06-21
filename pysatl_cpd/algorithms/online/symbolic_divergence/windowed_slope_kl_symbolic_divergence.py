# -*- coding: ascii -*-
"""
Windowed Slope + Kullback-Leibler Symbolic Divergence detector.

This module provides :class:`WindowedSlopeKLSymbolicDivergence`, the canonical
concrete windowed Symbolic Divergence detector combining the two-point
:class:`SlopeEncoder` with the :class:`KLDivergence`. Unlike
:class:`SlopeKLSymbolicDivergence`, which compares the running cumulative symbol
distribution against a reference fixed at the end of the learning period, this
variant compares a fixed-size recent window of symbols against a reference that
keeps growing as symbols leave that window.

Its configuration stores the component parameters (``delta``, ``gamma``,
``smoothing``) together with ``recent_window_size`` as scalars, so distinct
detector instances hash differently and are correctly identified by the
benchmarking framework. The change-point statistic defaults to the raw
divergence (``RawDivergenceStatistic``).
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass

from pysatl_cpd.algorithms.online.symbolic_divergence.divergences.kl_divergence import KLDivergence
from pysatl_cpd.algorithms.online.symbolic_divergence.encoders.slope_encoder import SlopeEncoder
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.base import IChangePointStatistic
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.raw import RawDivergenceStatistic
from pysatl_cpd.algorithms.online.symbolic_divergence.windowed_symbolic_divergence import (
    WindowedSymbolicDivergence,
    WindowedSymbolicDivergenceConfiguration,
    WindowedSymbolicDivergenceState,
)


@dataclass(kw_only=True, frozen=True)
class WindowedSlopeKLSymbolicDivergenceConfiguration(WindowedSymbolicDivergenceConfiguration):
    """
    Configuration for the windowed Slope + Kullback-Leibler Symbolic Divergence detector.

    The slope-encoder and divergence parameters are stored here as scalars so
    they participate in the configuration hash used to identify the detector.

    Attributes
    ----------
    delta
        Inner (dead-zone) threshold of the slope encoder.
    gamma
        Outer threshold of the slope encoder.
    smoothing
        Additive smoothing constant of the Kullback-Leibler divergence.

    Raises
    ------
    ValueError
        If ``learning_period_size`` or ``recent_window_size`` is not positive.
    """

    delta: float = 0.0
    gamma: float = 1.0
    smoothing: float = 1e-10

    def __repr__(self) -> str:
        """Return a short string representation of the configuration."""
        return (
            f"learning_period_size={self.learning_period_size}, "
            f"recent_window_size={self.recent_window_size}, "
            f"delta={self.delta}, gamma={self.gamma}, smoothing={self.smoothing}"
        )


@dataclass(kw_only=True, frozen=True)
class WindowedSlopeKLSymbolicDivergenceState(WindowedSymbolicDivergenceState):
    """State snapshot of the windowed Slope + Kullback-Leibler Symbolic Divergence detector."""


class WindowedSlopeKLSymbolicDivergence(
    WindowedSymbolicDivergence[WindowedSlopeKLSymbolicDivergenceConfiguration, WindowedSlopeKLSymbolicDivergenceState]
):
    """
    Windowed Symbolic Divergence detector using a slope encoder and KL divergence.

    This is the canonical concrete windowed detector: a two-point
    :class:`SlopeEncoder` feeds a :class:`KLDivergence`, the empirical
    distribution is taken over a fixed recent window, and the reference grows as
    symbols leave that window. The change-point statistic is chosen via the
    ``statistic`` argument (raw divergence by default).

    Parameters
    ----------
    learning_period_size
        Number of initial observations used to estimate the reference
        distribution. Must be strictly greater than the slope window size (2).
    recent_window_size
        Number of recent post-learning observations whose symbols form the
        empirical distribution. Must be strictly greater than the slope window
        size (2).
    delta
        Inner (dead-zone) threshold of the slope encoder. Must be non-negative.
    gamma
        Outer threshold of the slope encoder. Must be positive and strictly
        greater than ``delta``.
    smoothing
        Additive smoothing constant of the KL divergence. Must be non-negative.
    statistic
        Change-point statistic implementing ``IChangePointStatistic``. Defaults
        to ``RawDivergenceStatistic`` (the raw divergence).

    Notes
    -----
    Parameter validation is delegated: ``learning_period_size`` and
    ``recent_window_size`` are checked by the configuration and against the
    slope window size, while ``delta``, ``gamma``, and ``smoothing`` are
    validated by the slope encoder and the KL divergence. Each raises
    ``ValueError`` on invalid input.
    """

    def __init__(
        self,
        learning_period_size: int,
        recent_window_size: int,
        delta: float,
        gamma: float,
        smoothing: float = 1e-10,
        statistic: IChangePointStatistic | None = None,
    ) -> None:
        configuration = WindowedSlopeKLSymbolicDivergenceConfiguration(
            learning_period_size=learning_period_size,
            recent_window_size=recent_window_size,
            delta=delta,
            gamma=gamma,
            smoothing=smoothing,
        )
        super().__init__(
            configuration=configuration,
            encoder=SlopeEncoder(delta=delta, gamma=gamma),
            divergence=KLDivergence(smoothing=smoothing),
            statistic=statistic if statistic is not None else RawDivergenceStatistic(),
        )

    @property
    def name(self) -> str:
        """Human-readable algorithm name.

        Returns
        -------
        str
        """
        return "WindowedSlopeKLSymbolicDivergence"

    @property
    def configuration(self) -> WindowedSlopeKLSymbolicDivergenceConfiguration:
        """Current algorithm configuration.

        Returns
        -------
        WindowedSlopeKLSymbolicDivergenceConfiguration
        """
        return self._configuration

    @property
    def state(self) -> WindowedSlopeKLSymbolicDivergenceState:
        """Materialise an immutable state snapshot.

        Returns
        -------
        WindowedSlopeKLSymbolicDivergenceState
        """
        return WindowedSlopeKLSymbolicDivergenceState(
            is_in_learning_period=self.is_in_learning_period,
            samples_count=self.samples_count,
            symbol_counts=self.symbol_counts,
            reference_distribution=self.reference_distribution,
        )
