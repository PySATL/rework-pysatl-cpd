# -*- coding: ascii -*-
"""
Symbolic Divergence online change-point detection.

This package implements :class:`SymbolicDivergence`, a generalized online
change-point detector that encodes a sliding window of observations into a
symbol via a pluggable encoder (``h: R^k -> S``) and monitors the divergence
between the running empirical symbol distribution and a reference distribution
fixed at the end of the learning period. The divergence stream is then mapped
to the emitted change-point statistic by a pluggable statistic component.

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

Algorithm:

- ``SymbolicDivergence`` -- Generalized online detector combining a symbol
  encoder, a divergence, and a change-point statistic.
- ``SymbolicDivergenceConfiguration`` -- Configuration dataclass for
  ``SymbolicDivergence``.
- ``SymbolicDivergenceState`` -- State snapshot for ``SymbolicDivergence``.

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
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.symbolic_divergence.divergences import IDivergence, KLDivergence
from pysatl_cpd.algorithms.online.symbolic_divergence.encoders import ISymbolEncoder, SlopeEncoder
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics import (
    IChangePointStatistic,
    RawDivergenceStatistic,
    ScaledDivergenceStatistic,
)
from pysatl_cpd.algorithms.online.symbolic_divergence.symbolic_divergence import (
    SymbolicDivergence,
    SymbolicDivergenceConfiguration,
    SymbolicDivergenceState,
)

__all__ = [
    "IChangePointStatistic",
    "IDivergence",
    "ISymbolEncoder",
    "KLDivergence",
    "RawDivergenceStatistic",
    "ScaledDivergenceStatistic",
    "SlopeEncoder",
    "SymbolicDivergence",
    "SymbolicDivergenceConfiguration",
    "SymbolicDivergenceState",
]
