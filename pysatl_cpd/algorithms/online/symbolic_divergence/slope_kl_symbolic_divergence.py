# -*- coding: ascii -*-
"""
Slope + Kullback-Leibler Symbolic Divergence detector.

This module provides :class:`SlopeKLSymbolicDivergence`, the canonical concrete
Symbolic Divergence detector combining the two-point :class:`SlopeEncoder` with
the :class:`KLDivergence`. Its configuration stores the component parameters
(``delta``, ``gamma``, ``smoothing``) as scalars, so distinct detector instances
hash differently and are correctly identified by the benchmarking framework.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass

from pysatl_cpd.algorithms.online.symbolic_divergence.divergences.kl_divergence import KLDivergence
from pysatl_cpd.algorithms.online.symbolic_divergence.encoders.slope_encoder import SlopeEncoder
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.base import IChangePointStatistic
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.scaled import ScaledDivergenceStatistic
from pysatl_cpd.algorithms.online.symbolic_divergence.symbolic_divergence import (
    SymbolicDivergence,
    SymbolicDivergenceConfiguration,
    SymbolicDivergenceState,
)


@dataclass(kw_only=True, frozen=True)
class SlopeKLSymbolicDivergenceConfiguration(SymbolicDivergenceConfiguration):
    """
    Configuration for the Slope + Kullback-Leibler Symbolic Divergence detector.

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
        If ``learning_period_size`` is not positive.
    """

    delta: float = 0.0
    gamma: float = 1.0
    smoothing: float = 1e-10

    def __repr__(self) -> str:
        """Return a short string representation of the configuration."""
        return f"delta={self.delta}, gamma={self.gamma}, smoothing={self.smoothing}"


@dataclass(kw_only=True, frozen=True)
class SlopeKLSymbolicDivergenceState(SymbolicDivergenceState):
    """State snapshot of the Slope + Kullback-Leibler Symbolic Divergence detector."""


class SlopeKLSymbolicDivergence(
    SymbolicDivergence[SlopeKLSymbolicDivergenceConfiguration, SlopeKLSymbolicDivergenceState]
):
    """
    Symbolic Divergence detector using a slope encoder and KL divergence.

    This is the canonical concrete detector: a two-point :class:`SlopeEncoder`
    feeds a :class:`KLDivergence`, with the change-point statistic chosen via the
    ``statistic`` argument (raw divergence by default).

    Parameters
    ----------
    learning_period_size
        Number of initial observations used to estimate the reference
        distribution. Must be strictly greater than the slope window size (2).
    delta
        Inner (dead-zone) threshold of the slope encoder. Must be non-negative.
    gamma
        Outer threshold of the slope encoder. Must be positive and strictly
        greater than ``delta``.
    smoothing
        Additive smoothing constant of the KL divergence. Must be non-negative.
    statistic
        Change-point statistic implementing ``IChangePointStatistic``. Defaults
        to ``ScaledDivergenceStatistic`` (the ``2 n D`` statistic).

    Notes
    -----
    Parameter validation is delegated: ``learning_period_size`` is checked by
    the configuration and against the slope window size, while ``delta``,
    ``gamma``, and ``smoothing`` are validated by the slope encoder and the KL
    divergence. Each raises ``ValueError`` on invalid input.
    """

    def __init__(
        self,
        learning_period_size: int,
        delta: float,
        gamma: float,
        smoothing: float = 1e-10,
        statistic: IChangePointStatistic | None = None,
    ) -> None:
        configuration = SlopeKLSymbolicDivergenceConfiguration(
            learning_period_size=learning_period_size,
            delta=delta,
            gamma=gamma,
            smoothing=smoothing,
        )
        super().__init__(
            configuration=configuration,
            encoder=SlopeEncoder(delta=delta, gamma=gamma),
            divergence=KLDivergence(smoothing=smoothing),
            statistic=statistic if statistic is not None else ScaledDivergenceStatistic(),
        )

    @property
    def name(self) -> str:
        """Human-readable algorithm name.

        Returns
        -------
        str
        """
        return "SlopeKLSymbolicDivergence"

    @property
    def configuration(self) -> SlopeKLSymbolicDivergenceConfiguration:
        """Current algorithm configuration.

        Returns
        -------
        SlopeKLSymbolicDivergenceConfiguration
        """
        return self._configuration

    @property
    def state(self) -> SlopeKLSymbolicDivergenceState:
        """Materialise an immutable state snapshot.

        Returns
        -------
        SlopeKLSymbolicDivergenceState
        """
        return SlopeKLSymbolicDivergenceState(
            is_in_learning_period=self.is_in_learning_period,
            samples_count=self.samples_count,
            symbol_counts=self.symbol_counts,
            reference_distribution=self.reference_distribution,
        )
