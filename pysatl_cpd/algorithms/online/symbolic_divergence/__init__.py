# -*- coding: ascii -*-
"""
Symbolic Divergence online change-point detection.

This package implements :class:`SymbolicDivergence`, a generalized online
change-point detector that encodes a sliding window of observations into a
symbol via a pluggable encoder (``h: R^k -> S``) and monitors the divergence
between the running empirical symbol distribution and a reference distribution
fixed at the end of the learning period. The divergence stream is then mapped
to the emitted change-point statistic by a pluggable statistic component.

A windowed sibling, :class:`WindowedSymbolicDivergence`, instead compares a
fixed-size recent window of symbols against a reference that keeps growing as
symbols leave that window.

The exported class implements the ``OnlineAlgorithm`` interface from
``pysatl_cpd.core.online`` and is compatible with ``OnlineResetDetector``,
``OnlineCpdSolver``, runtime wrappers, and the benchmarking framework.

The method is parameterized by three interchangeable components, so the
Slope-encoding + Kullback-Leibler + ``2 n D`` combination is a single special
case of a broader family:

- a symbol encoder implementing ``ISymbolEncoder`` (the ``SlopeEncoder`` with
  ``k = 2`` is the canonical special case),
- a divergence function implementing ``IDivergence`` (Kullback-Leibler being
  one option),
- a change-point statistic implementing ``IChangePointStatistic``
  (``RawDivergenceStatistic`` by default, ``ScaledDivergenceStatistic`` for the
  ``2 n D`` statistic).

.. raw:: html

    <h2>Public API</h2>

Algorithms:

- ``SymbolicDivergence`` -- Abstract, generic online detector combining a
  symbol encoder, a divergence, and a change-point statistic. Subclass it to
  build concrete detectors.
- ``SymbolicDivergenceConfiguration`` -- Base configuration dataclass.
- ``SymbolicDivergenceState`` -- Base state snapshot.
- ``SlopeKLSymbolicDivergence`` -- Concrete detector using a slope encoder and
  Kullback-Leibler divergence; stores component parameters in its
  configuration.
- ``SlopeKLSymbolicDivergenceConfiguration`` -- Configuration for
  ``SlopeKLSymbolicDivergence``.
- ``SlopeKLSymbolicDivergenceState`` -- State snapshot for
  ``SlopeKLSymbolicDivergence``.
- ``WindowedSymbolicDivergence`` -- Abstract, generic windowed detector
  comparing a fixed recent window of symbols against a growing reference.
- ``WindowedSymbolicDivergenceConfiguration`` -- Base configuration dataclass
  for the windowed detector (adds ``recent_window_size``).
- ``WindowedSymbolicDivergenceState`` -- Base state snapshot for the windowed
  detector.
- ``WindowedSlopeKLSymbolicDivergence`` -- Concrete windowed detector using a
  slope encoder and Kullback-Leibler divergence.
- ``WindowedSlopeKLSymbolicDivergenceConfiguration`` -- Configuration for
  ``WindowedSlopeKLSymbolicDivergence``.
- ``WindowedSlopeKLSymbolicDivergenceState`` -- State snapshot for
  ``WindowedSlopeKLSymbolicDivergence``.

Encoder components (from ``encoders``):

- ``ISymbolEncoder`` -- Protocol for window-to-symbol encoders.
- ``SlopeEncoder`` -- Two-point slope encoder (``k = 2``).

Divergence components (from ``divergences``):

- ``IDivergence`` -- Protocol for divergence functions.
- ``KLDivergence`` -- Kullback-Leibler divergence with smoothing.

Change-point statistic components (from ``statistics``):

- ``IChangePointStatistic`` -- Protocol for divergence-to-statistic mappings.
- ``RawDivergenceStatistic`` -- Identity statistic (default).
- ``ScaledDivergenceStatistic`` -- ``scale * sample_size * divergence``.
- ``LogScaledDivergenceStatistic`` -- ``scale * sample_size /
  log(sample_size + 1) * divergence``.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.symbolic_divergence.divergences import IDivergence, KLDivergence
from pysatl_cpd.algorithms.online.symbolic_divergence.encoders import ISymbolEncoder, SlopeEncoder
from pysatl_cpd.algorithms.online.symbolic_divergence.slope_kl_symbolic_divergence import (
    SlopeKLSymbolicDivergence,
    SlopeKLSymbolicDivergenceConfiguration,
    SlopeKLSymbolicDivergenceState,
)
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics import (
    IChangePointStatistic,
    LogScaledDivergenceStatistic,
    RawDivergenceStatistic,
    ScaledDivergenceStatistic,
)
from pysatl_cpd.algorithms.online.symbolic_divergence.symbolic_divergence import (
    SymbolicDivergence,
    SymbolicDivergenceConfiguration,
    SymbolicDivergenceState,
)
from pysatl_cpd.algorithms.online.symbolic_divergence.windowed_slope_kl_symbolic_divergence import (
    WindowedSlopeKLSymbolicDivergence,
    WindowedSlopeKLSymbolicDivergenceConfiguration,
    WindowedSlopeKLSymbolicDivergenceState,
)
from pysatl_cpd.algorithms.online.symbolic_divergence.windowed_symbolic_divergence import (
    WindowedSymbolicDivergence,
    WindowedSymbolicDivergenceConfiguration,
    WindowedSymbolicDivergenceState,
)

__all__ = [
    "IChangePointStatistic",
    "IDivergence",
    "ISymbolEncoder",
    "KLDivergence",
    "LogScaledDivergenceStatistic",
    "RawDivergenceStatistic",
    "ScaledDivergenceStatistic",
    "SlopeEncoder",
    "SlopeKLSymbolicDivergence",
    "SlopeKLSymbolicDivergenceConfiguration",
    "SlopeKLSymbolicDivergenceState",
    "SymbolicDivergence",
    "SymbolicDivergenceConfiguration",
    "SymbolicDivergenceState",
    "WindowedSlopeKLSymbolicDivergence",
    "WindowedSlopeKLSymbolicDivergenceConfiguration",
    "WindowedSlopeKLSymbolicDivergenceState",
    "WindowedSymbolicDivergence",
    "WindowedSymbolicDivergenceConfiguration",
    "WindowedSymbolicDivergenceState",
]
