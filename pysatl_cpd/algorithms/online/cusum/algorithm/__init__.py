# -*- coding: ascii -*-
"""
CUSUM algorithm implementations.

This subpackage provides concrete online change-point detection algorithms
based on the Cumulative Sum (CUSUM) family. All implementations inherit from
the ``GeneralizedCUSUM`` base class and conform to the ``OnlineAlgorithm``
interface, making them compatible with detectors, wrappers, and the
benchmarking framework in ``pysatl_cpd.core.online`` and
``pysatl_cpd.benchmark``.

.. raw:: html

    <h2>Public API</h2>

- ``AutoregressiveCUSUM`` -- CUSUM detector for univariate autoregressive
  Gaussian time series. Fits an AR model during the learning period and
  applies Page's change-point function to the residuals.
- ``AutoregressiveCusumConfiguration`` -- Configuration dataclass for
  ``AutoregressiveCUSUM`` (``delta``, ``autoreg_order``, ``autoreg_window``).
- ``AutoregressiveCusumState`` -- Immutable state snapshot for
  ``AutoregressiveCUSUM``.
- ``CrosierCusum`` -- CUSUM detector based on the Crosier statistic with
  norm-based shrinkage for multivariate Gaussian observations.
- ``CrosierCusumConfiguration`` -- Configuration dataclass for
  ``CrosierCusum`` (``delta``, ``cov_reg``).
- ``CrosierCusumState`` -- Immutable state snapshot for ``CrosierCusum``.
- ``PageTwoSidedCusum`` -- Two-sided Page CUSUM detector for Gaussian
  observations, tracking both positive and negative mean shifts.
- ``PageTwoSidedCusumConfiguration`` -- Configuration dataclass for
  ``PageTwoSidedCusum`` (``delta``, ``cov_reg``).
- ``PageTwoSidedCusumState`` -- Immutable state snapshot for
  ``PageTwoSidedCusum``.
- ``VarianceTwoSidedCUSUM`` -- Two-sided CUSUM detector focused on variance
  (scale) changes in univariate Gaussian data.
- ``VarianceTwoSidedCusumConfiguration`` -- Configuration dataclass for
  ``VarianceTwoSidedCUSUM`` (``delta``).
- ``VarianceTwoSidedCusumState`` -- Immutable state snapshot for
  ``VarianceTwoSidedCUSUM``.

.. raw:: html

    <h2>Examples</h2>

Instantiate a CUSUM algorithm and process observations one at a time::

    >>> import numpy as np
    >>> from pysatl_cpd.algorithms.online.cusum.algorithm import PageTwoSidedCusum
    >>> algo = PageTwoSidedCusum(learning_period_size=30, delta=0.5)
    >>> rng = np.random.default_rng(42)
    >>> pre_change = rng.normal(loc=0.0, scale=1.0, size=30)
    >>> post_change = rng.normal(loc=2.0, scale=1.0, size=20)
    >>> for x in pre_change:
    ...     _ = algo.process(np.array([x]))
    >>> for x in post_change:
    ...     statistic = algo.process(np.array([x]))
    >>> print(f"Final statistic: {statistic:.4f}")
    Final statistic: ...

Use a CUSUM algorithm with ``OnlineResetDetector`` for threshold-based
detection::

    >>> from pysatl_cpd.algorithms.online.cusum.algorithm import CrosierCusum
    >>> from pysatl_cpd.core.online import OnlineResetDetector
    >>> from pysatl_cpd.data.generator import preset_dataset
    >>> algo = CrosierCusum(learning_period_size=30, delta=0.5)
    >>> detector = OnlineResetDetector(algo, threshold=2.0)
    >>> dataset = preset_dataset("mean_shifts", n_series=1, seed=7, series_length=180)
    >>> provider = dataset[0]
    >>> trace = detector.detect(provider)
    >>> print(f"Detected change points: {trace.detected_change_points}")
    Detected change points: ...

Compare multiple CUSUM variants in a no-reset benchmark::

    >>> from pysatl_cpd.algorithms.online.cusum.algorithm import (
    ...     PageTwoSidedCusum,
    ...     VarianceTwoSidedCUSUM,
    ... )
    >>> from pysatl_cpd.benchmark.online.noreset import (
    ...     NoResetOnlineDetector,
    ...     OnlineNoResetBenchmarkEntry,
    ... )
    >>> from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import AutoThresholdsRange
    >>> page = PageTwoSidedCusum(learning_period_size=50, delta=0.5)
    >>> var = VarianceTwoSidedCUSUM(learning_period_size=50, delta=0.5)
    >>> page_entry = OnlineNoResetBenchmarkEntry(
    ...     algorithm=page, thresholds=AutoThresholdsRange(count=50)
    ... )
    >>> var_entry = OnlineNoResetBenchmarkEntry(
    ...     algorithm=var, thresholds=AutoThresholdsRange(count=50)
    ... )

.. raw:: html

    <h2>Notes</h2>

- All algorithms require a ``learning_period_size`` of initial observations
  for parameter estimation before producing non-zero statistics.
- ``AutoregressiveCUSUM`` and ``VarianceTwoSidedCUSUM`` accept only
  univariate observations (shape ``(1,)``). ``CrosierCusum`` and
  ``PageTwoSidedCusum`` support multivariate input.
- The ``delta`` parameter controls sensitivity: larger values reduce
  false alarms but increase detection delay.
- These classes are also re-exported by the parent ``cusum`` package
  (``pysatl_cpd.algorithms.online.cusum``) and the top-level
  ``pysatl_cpd.algorithms.online`` package.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.cusum.algorithm.autoregressive_cusum import (
    AutoregressiveCUSUM,
    AutoregressiveCusumConfiguration,
    AutoregressiveCusumState,
)
from pysatl_cpd.algorithms.online.cusum.algorithm.crosier_cusum import (
    CrosierCusum,
    CrosierCusumConfiguration,
    CrosierCusumState,
)
from pysatl_cpd.algorithms.online.cusum.algorithm.page_cusum import (
    PageTwoSidedCusum,
    PageTwoSidedCusumConfiguration,
    PageTwoSidedCusumState,
)
from pysatl_cpd.algorithms.online.cusum.algorithm.variance_cusum import (
    VarianceTwoSidedCUSUM,
    VarianceTwoSidedCusumConfiguration,
    VarianceTwoSidedCusumState,
)

__all__ = [
    "AutoregressiveCUSUM",
    "AutoregressiveCusumConfiguration",
    "AutoregressiveCusumState",
    "CrosierCusum",
    "CrosierCusumConfiguration",
    "CrosierCusumState",
    "PageTwoSidedCusum",
    "PageTwoSidedCusumConfiguration",
    "PageTwoSidedCusumState",
    "VarianceTwoSidedCUSUM",
    "VarianceTwoSidedCusumConfiguration",
    "VarianceTwoSidedCusumState",
]
