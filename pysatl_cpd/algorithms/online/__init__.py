# -*- coding: ascii -*-
"""
Online change-point detection algorithms.

This package is the primary import surface for all online change-point
detection algorithm implementations. Every exported class implements the
``OnlineAlgorithm`` interface from ``pysatl_cpd.core.online``, making it
compatible with ``OnlineResetDetector``, ``OnlineCpdSolver``, runtime
wrappers, and the benchmarking framework.

The package re-exports concrete algorithms and their associated
configuration/state dataclasses from four subpackages:

- ``bayesian`` -- Bayesian online change-point detection (BOCPD) following
  the Adams and MacKay (2007) message-passing framework. Includes an
  abstract base class, a ready-to-use univariate Gaussian conjugate
  detector, component factory functions, and a CPF-type enum. See the
  :mod:`~pysatl_cpd.algorithms.online.bayesian` docstring for details.
- ``control_charts`` -- Statistical process control chart techniques.
  Currently provides the Shewhart control chart with a sliding-window
  statistic. See the
  :mod:`~pysatl_cpd.algorithms.online.control_charts` docstring for
  details.
- ``cusum`` -- A family of Cumulative Sum (CUSUM) detectors including
  Page's two-sided CUSUM, the Crosier multivariate statistic, variance
  change detection, and autoregressive time series detection. See the
  :mod:`~pysatl_cpd.algorithms.online.cusum` docstring for details.
- ``symbolic_divergence`` -- A generalized symbolic-encoding detector that
  monitors the divergence between the running empirical symbol distribution
  and a reference distribution, with pluggable encoder, divergence, and
  change-point statistic components. A windowed variant compares a fixed
  recent window of symbols against a growing reference. See the
  :mod:`~pysatl_cpd.algorithms.online.symbolic_divergence` docstring for
  details.

The ``classification`` directory is an internal implementation detail and
is not part of the public API.

.. raw:: html

    <h2>Public API</h2>

Bayesian algorithms (from ``bayesian``):

- ``AbstractBayesian`` -- Abstract base class implementing the BOCPD
  message-passing loop.
- ``BayesianCPFType`` -- String enum for change-point score function
  selection (``MAX_RUN_LENGTH`` or ``DROP``).
- ``BayesianOnlineCPDConfiguration`` -- Base configuration dataclass shared
  by all Bayesian online detectors.
- ``BayesianOnlineCPDState`` -- Base state dataclass capturing the BOCPD
  run-length posterior snapshot.
- ``UnivariateGaussianConjugateBOCPD`` -- Concrete BOCPD detector for
  univariate data with a Normal-Inverse-Gamma conjugate prior.
- ``UnivariateGaussianConjugateBOCPDConfiguration`` -- Configuration
  dataclass for the univariate Gaussian conjugate detector.
- ``UnivariateGaussianConjugateBOCPDState`` -- State dataclass for the
  univariate Gaussian conjugate detector.

Control chart algorithms (from ``control_charts``):

- ``ShewhartControlChart`` -- Shewhart control chart with a sliding-window
  statistic for mean-shift detection.
- ``ShewhartControlChartConfiguration`` -- Configuration dataclass for the
  Shewhart chart.
- ``ShewhartControlChartState`` -- State snapshot for the Shewhart chart.

CUSUM algorithms (from ``cusum``):

- ``PageTwoSidedCusum`` -- Two-sided Page CUSUM for Gaussian mean-shift
  detection (univariate and multivariate).
- ``PageTwoSidedCusumConfiguration`` -- Configuration dataclass for
  ``PageTwoSidedCusum``.
- ``PageTwoSidedCusumState`` -- State snapshot for ``PageTwoSidedCusum``.
- ``CrosierCusum`` -- Multivariate CUSUM based on the Crosier statistic
  with norm-based shrinkage.
- ``CrosierCusumConfiguration`` -- Configuration dataclass for
  ``CrosierCusum``.
- ``CrosierCusumState`` -- State snapshot for ``CrosierCusum``.
- ``VarianceTwoSidedCUSUM`` -- Two-sided CUSUM for univariate variance
  (scale) change detection.
- ``VarianceTwoSidedCusumConfiguration`` -- Configuration dataclass for
  ``VarianceTwoSidedCUSUM``.
- ``VarianceTwoSidedCusumState`` -- State snapshot for
  ``VarianceTwoSidedCUSUM``.
- ``AutoregressiveCUSUM`` -- CUSUM for univariate autoregressive Gaussian
  time series using Page's CPF on AR model residuals.
- ``AutoregressiveCusumConfiguration`` -- Configuration dataclass for
  ``AutoregressiveCUSUM``.
- ``AutoregressiveCusumState`` -- State snapshot for ``AutoregressiveCUSUM``.

Symbolic Divergence algorithms (from ``symbolic_divergence``):

- ``SymbolicDivergence`` -- Abstract, generic base detector combining a symbol
  encoder, a divergence, and a change-point statistic.
- ``SymbolicDivergenceConfiguration`` -- Base configuration dataclass.
- ``SymbolicDivergenceState`` -- Base state snapshot.
- ``SlopeKLSymbolicDivergence`` -- Concrete slope-encoder + KL-divergence
  detector storing component parameters in its configuration.
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
- ``WindowedSlopeKLSymbolicDivergence`` -- Concrete windowed slope-encoder +
  KL-divergence detector.
- ``WindowedSlopeKLSymbolicDivergenceConfiguration`` -- Configuration for
  ``WindowedSlopeKLSymbolicDivergence``.
- ``WindowedSlopeKLSymbolicDivergenceState`` -- State snapshot for
  ``WindowedSlopeKLSymbolicDivergence``.
- ``ISymbolEncoder`` -- Protocol for window-to-symbol encoders.
- ``SlopeEncoder`` -- Two-point slope encoder (``k = 2``).
- ``IDivergence`` -- Protocol for divergence functions.
- ``KLDivergence`` -- Kullback-Leibler divergence with smoothing.
- ``IChangePointStatistic`` -- Protocol for divergence-to-statistic mappings.
- ``RawDivergenceStatistic`` -- Identity change-point statistic (default).
- ``ScaledDivergenceStatistic`` -- ``scale * sample_size * divergence``
  statistic (``scale = 2.0`` reproduces ``2 n D``).
- ``LogScaledDivergenceStatistic`` -- ``scale * sample_size /
  log(sample_size + 1) * divergence`` statistic.

Notes
-----
All algorithms require a ``learning_period_size`` of initial observations
for parameter estimation before producing non-zero detection statistics.

``AutoregressiveCUSUM`` and ``VarianceTwoSidedCUSUM`` accept only
univariate observations. ``CrosierCusum``, ``PageTwoSidedCusum``,
``ShewhartControlChart``, and ``UnivariateGaussianConjugateBOCPD`` support
multivariate input (though the BOCPD detector is designed for univariate
data).

``AutoregressiveCUSUM`` requires the optional ``arch`` package. Install
it via ``poetry add arch``.

For detailed mathematical formulations, component architecture, and
extended usage examples, consult the docstrings of the individual
subpackages: ``bayesian``, ``control_charts``, and ``cusum``.

Examples
--------
Import and use a Shewhart control chart with ``OnlineResetDetector``:

>>> import numpy as np
>>> from pysatl_cpd.algorithms.online import ShewhartControlChart
>>> from pysatl_cpd.core.online import OnlineResetDetector
>>> from pysatl_cpd.data.generator import preset_dataset
>>> dataset = preset_dataset("mean_shifts", n_series=1, seed=7, series_length=180, n_features=1)
>>> provider = dataset[0]
>>> detector = OnlineResetDetector(
...     ShewhartControlChart(learning_period_size=30, window_size=10),
...     threshold=2.0,
... )
>>> trace = detector.detect(provider)
>>> len(trace.detected_change_points) > 0
True

Run a Bayesian BOCPD detector on a synthetic series:

>>> from pysatl_cpd.algorithms.online import UnivariateGaussianConjugateBOCPD
>>> algo = UnivariateGaussianConjugateBOCPD(
...     learning_period_size=20,
...     hazard_lambda=50.0,
... )
>>> rng = np.random.default_rng(42)
>>> series = np.concatenate([rng.normal(0.0, 1.0, 30), rng.normal(3.0, 1.0, 30)])
>>> scores = [algo.process(x) for x in series]
>>> all(s == 0.0 for s in scores[:20])
True

Run a CUSUM detector and inspect its state:

>>> from pysatl_cpd.algorithms.online import PageTwoSidedCusum
>>> algo = PageTwoSidedCusum(learning_period_size=10, delta=0.5)
>>> for v in [0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.1, -0.3, 0.0, 0.1, 2.0, 2.5]:
...     stat = algo.process(np.array([v]))
>>> algo.state.t
12
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.bayesian import (
    AbstractBayesian,
    BayesianCPFType,
    BayesianOnlineCPDConfiguration,
    BayesianOnlineCPDState,
    UnivariateGaussianConjugateBOCPD,
    UnivariateGaussianConjugateBOCPDConfiguration,
    UnivariateGaussianConjugateBOCPDState,
)
from pysatl_cpd.algorithms.online.control_charts.shewhart_control_chart import (
    ShewhartControlChart,
    ShewhartControlChartConfiguration,
    ShewhartControlChartState,
)
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
from pysatl_cpd.algorithms.online.symbolic_divergence import (
    IChangePointStatistic,
    IDivergence,
    ISymbolEncoder,
    KLDivergence,
    LogScaledDivergenceStatistic,
    RawDivergenceStatistic,
    ScaledDivergenceStatistic,
    SlopeEncoder,
    SlopeKLSymbolicDivergence,
    SlopeKLSymbolicDivergenceConfiguration,
    SlopeKLSymbolicDivergenceState,
    SymbolicDivergence,
    SymbolicDivergenceConfiguration,
    SymbolicDivergenceState,
    WindowedSlopeKLSymbolicDivergence,
    WindowedSlopeKLSymbolicDivergenceConfiguration,
    WindowedSlopeKLSymbolicDivergenceState,
    WindowedSymbolicDivergence,
    WindowedSymbolicDivergenceConfiguration,
    WindowedSymbolicDivergenceState,
)

__all__ = [
    "AbstractBayesian",
    "AutoregressiveCUSUM",
    "AutoregressiveCusumConfiguration",
    "AutoregressiveCusumState",
    "BayesianCPFType",
    "BayesianOnlineCPDConfiguration",
    "BayesianOnlineCPDState",
    "CrosierCusum",
    "CrosierCusumConfiguration",
    "CrosierCusumState",
    "IChangePointStatistic",
    "IDivergence",
    "ISymbolEncoder",
    "KLDivergence",
    "LogScaledDivergenceStatistic",
    "PageTwoSidedCusum",
    "PageTwoSidedCusumConfiguration",
    "PageTwoSidedCusumState",
    "RawDivergenceStatistic",
    "ScaledDivergenceStatistic",
    "ShewhartControlChart",
    "ShewhartControlChartConfiguration",
    "ShewhartControlChartState",
    "SlopeEncoder",
    "SlopeKLSymbolicDivergence",
    "SlopeKLSymbolicDivergenceConfiguration",
    "SlopeKLSymbolicDivergenceState",
    "SymbolicDivergence",
    "SymbolicDivergenceConfiguration",
    "SymbolicDivergenceState",
    "UnivariateGaussianConjugateBOCPD",
    "UnivariateGaussianConjugateBOCPDConfiguration",
    "UnivariateGaussianConjugateBOCPDState",
    "VarianceTwoSidedCUSUM",
    "VarianceTwoSidedCusumConfiguration",
    "VarianceTwoSidedCusumState",
    "WindowedSlopeKLSymbolicDivergence",
    "WindowedSlopeKLSymbolicDivergenceConfiguration",
    "WindowedSlopeKLSymbolicDivergenceState",
    "WindowedSymbolicDivergence",
    "WindowedSymbolicDivergenceConfiguration",
    "WindowedSymbolicDivergenceState",
]
