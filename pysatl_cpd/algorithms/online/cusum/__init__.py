# -*- coding: ascii -*-
"""
CUSUM algorithms for online change-point detection.

This package provides a family of Cumulative Sum (CUSUM) detectors for
online change-point detection. All exported algorithms implement the
``OnlineAlgorithm`` interface from ``pysatl_cpd.core.online``, making them
compatible with ``OnlineResetDetector``, the no-reset benchmarking
framework, and the visualization layer.

The package is organized into subpackages that separate concerns:

- ``abstracts`` -- Abstract base classes and protocols. Defines
  ``GeneralizedCUSUM`` and the component interfaces (``IEstimatingSchema``,
  ``IMonitoringSchema``, ``ICusumChangepointFunc``). See the
  :mod:`~pysatl_cpd.algorithms.online.cusum.abstracts` docstring for
  details.
- ``algorithm`` -- Concrete CUSUM algorithm implementations. This is where
  ``PageTwoSidedCusum``, ``CrosierCusum``, ``VarianceTwoSidedCUSUM``, and
  ``AutoregressiveCUSUM`` live. See the
  :mod:`~pysatl_cpd.algorithms.online.cusum.algorithm` docstring for
  details.
- ``component`` -- Building blocks (change-point functions, estimators,
  monitoring schemas) wired together by the concrete algorithms. See the
  :mod:`~pysatl_cpd.algorithms.online.cusum.component` docstring for
  details.
- ``utils`` -- Internal factory helpers for observation normalization.
  See the :mod:`~pysatl_cpd.algorithms.online.cusum.utils` docstring for
  details.

.. raw:: html

    <h2>Public API</h2>

- ``PageTwoSidedCusum`` -- Two-sided Page CUSUM for Gaussian mean-shift
  detection. Supports univariate and multivariate observations.
- ``PageTwoSidedCusumConfiguration`` -- Configuration dataclass for
  ``PageTwoSidedCusum`` (``delta``, ``cov_reg``, ``adaptive_estimation``).
- ``PageTwoSidedCusumState`` -- Immutable state snapshot for
  ``PageTwoSidedCusum``.
- ``CrosierCusum`` -- Multivariate CUSUM based on the Crosier statistic
  with norm-based shrinkage.
- ``CrosierCusumConfiguration`` -- Configuration dataclass for
  ``CrosierCusum`` (``delta``, ``cov_reg``, ``adaptive_estimation``).
- ``CrosierCusumState`` -- Immutable state snapshot for ``CrosierCusum``.
- ``VarianceTwoSidedCUSUM`` -- Two-sided CUSUM for univariate variance
  (scale) change detection.
- ``VarianceTwoSidedCusumConfiguration`` -- Configuration dataclass for
  ``VarianceTwoSidedCUSUM`` (``delta``, ``adaptive_estimation``).
- ``VarianceTwoSidedCusumState`` -- Immutable state snapshot for
  ``VarianceTwoSidedCUSUM``.
- ``AutoregressiveCUSUM`` -- CUSUM for univariate autoregressive Gaussian
  time series. Fits an AR model during the learning period and applies
  Page's CPF to the residuals.
- ``AutoregressiveCusumConfiguration`` -- Configuration dataclass for
  ``AutoregressiveCUSUM`` (``delta``, ``autoreg_order``,
  ``autoreg_window``, ``adaptive_estimation``).
- ``AutoregressiveCusumState`` -- Immutable state snapshot for
  ``AutoregressiveCUSUM``.

Notes
-----
All algorithms require a ``learning_period_size`` of initial observations
for parameter estimation before producing non-zero statistics.

``AutoregressiveCUSUM`` and ``VarianceTwoSidedCUSUM`` accept only
univariate observations (shape ``(1,)``). ``CrosierCusum`` and
``PageTwoSidedCusum`` support multivariate input.

The ``delta`` parameter controls sensitivity: larger values reduce
false alarms but increase detection delay.

``AutoregressiveCUSUM`` requires the optional ``arch`` package. Install
it via ``poetry add arch``.

These classes are also re-exported by the parent
``pysatl_cpd.algorithms.online`` package.

Examples
--------
Instantiate a CUSUM algorithm and process observations one at a time:

>>> import numpy as np
>>> from pysatl_cpd.algorithms.online.cusum import PageTwoSidedCusum
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
detection:

>>> from pysatl_cpd.algorithms.online.cusum import CrosierCusum
>>> from pysatl_cpd.core.online import OnlineResetDetector
>>> from pysatl_cpd.data.generator import preset_dataset
>>> algo = CrosierCusum(learning_period_size=30, delta=0.5)
>>> detector = OnlineResetDetector(algo, threshold=2.0)
>>> dataset = preset_dataset("mean_shifts", n_series=1, seed=7, series_length=180)
>>> provider = dataset[0]
>>> trace = detector.detect(provider)
>>> print(f"Detected change points: {trace.detected_change_points}")
Detected change points: ...

Compare multiple CUSUM variants in a no-reset benchmark:

>>> from pysatl_cpd.algorithms.online.cusum import (
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
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.cusum.algorithm import (
    AutoregressiveCUSUM,
    AutoregressiveCusumConfiguration,
    AutoregressiveCusumState,
    CrosierCusum,
    CrosierCusumConfiguration,
    CrosierCusumState,
    PageTwoSidedCusum,
    PageTwoSidedCusumConfiguration,
    PageTwoSidedCusumState,
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
