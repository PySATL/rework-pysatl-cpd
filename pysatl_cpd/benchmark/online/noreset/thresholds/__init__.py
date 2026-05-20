# -*- coding: ascii -*-
"""Threshold range types for no-reset benchmarks.

This module provides strategy objects that define how thresholds are
generated or resolved during no-reset benchmark campaigns. Each range
type produces a sequence of threshold values that are swept over
continuous detection traces to evaluate detector behavior at different
operating points.

.. raw:: html

    <h2>Public API</h2>

- ``AutoThresholdsRange`` -- Placeholder range whose thresholds are
  resolved later by an external analyzer (e.g., ``NoResetThresholdResolver``
  in the ``resolver`` submodule). Stores only a desired count.
- ``LinearThresholdsRange`` -- Generates a linearly spaced grid of
  thresholds between a start and end value using ``numpy.linspace``.
- ``ManualThresholdsRange`` -- Uses an explicit user-supplied list of
  threshold values as-is.
- ``ThresholdsRange`` -- Abstract base class for all range strategies.
  Subclasses populate ``thresholds_range`` in ``__post_init__``.

.. raw:: html

    <h2>Submodules</h2>

- ``ranges`` -- Defines the four range classes listed above. See its
  module docstring for class-level details.
- ``resolver`` -- Contains ``NoResetThresholdResolver`` for converting
  range specifications into concrete threshold lists, plus
  ``auto_pick_thresholds`` and supporting configuration/result types
  for data-driven threshold selection. See its module docstring for
  details.

Notes
-----
All range classes are dataclasses that populate ``thresholds_range``
during ``__post_init__``. The base ``ThresholdsRange`` is abstract
and cannot be instantiated directly.

``AutoThresholdsRange`` leaves ``thresholds_range`` empty; it is meant
to be consumed by ``NoResetThresholdResolver``, which infers bounds
from detection function values.

``LinearThresholdsRange`` requires ``count >= 1``; ``AutoThresholdsRange``
enforces the same constraint on its count.

Examples
--------
Create a linear threshold sweep for a benchmark entry:

>>> from pysatl_cpd.algorithms.online import ShewhartControlChart
>>> from pysatl_cpd.benchmark.online.noreset import OnlineNoResetBenchmarkEntry
>>> from pysatl_cpd.benchmark.online.noreset.thresholds import LinearThresholdsRange
>>> from pysatl_cpd.benchmark.online.noreset.tooling.bisegment_cut import BisegmentCut
>>> entry = OnlineNoResetBenchmarkEntry(
...     algorithm=ShewhartControlChart(learning_period_size=20, window_size=10),
...     thresholds=LinearThresholdsRange(start=1.5, end=3.0, count=6),
...     bisegment_cut=BisegmentCut.parse((8, 0)),
... )
>>> list(entry.thresholds.thresholds_range)
[np.float64(1.5), np.float64(1.8), np.float64(2.1), np.float64(2.4), np.float64(2.7), np.float64(3.0)]

Define thresholds explicitly with ``ManualThresholdsRange``:

>>> from pysatl_cpd.benchmark.online.noreset.thresholds import ManualThresholdsRange
>>> manual = ManualThresholdsRange(_values=[0.5, 1.0, 2.0, 5.0])
>>> list(manual.thresholds_range)
[0.5, 1.0, 2.0, 5.0]

Defer threshold selection with ``AutoThresholdsRange`` for later
resolution by ``NoResetThresholdResolver``:

>>> from pysatl_cpd.benchmark.online.noreset.thresholds import AutoThresholdsRange
>>> auto = AutoThresholdsRange(count=5)
>>> auto.thresholds_range
()
>>> auto.count
5
"""

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import (
    AutoThresholdsRange,
    LinearThresholdsRange,
    ManualThresholdsRange,
    ThresholdsRange,
)

__all__ = [
    "AutoThresholdsRange",
    "LinearThresholdsRange",
    "ManualThresholdsRange",
    "ThresholdsRange",
]
