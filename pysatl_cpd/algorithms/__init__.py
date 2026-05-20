# -*- coding: ascii -*-
"""Online algorithms for change-point detection.

This module is the top-level entry point for all online change-point detection
algorithm implementations. It re-exports every public class and dataclass from
the ``pysatl_cpd.algorithms.online`` subpackage, providing a single import
surface for Bayesian, CUSUM, and control chart detectors.

All exported algorithms implement the ``OnlineAlgorithm`` interface from
``pysatl_cpd.core.online``, making them compatible with ``OnlineResetDetector``,
``OnlineCpdSolver``, runtime wrappers, and the benchmarking framework.

.. raw:: html

    <h2>Subpackages</h2>

- ``online`` -- The sole subpackage containing all algorithm implementations.
  Organized into three family subpackages: ``bayesian`` (BOCPD message-passing),
  ``cusum`` (Cumulative Sum detectors), and ``control_charts`` (Shewhart
  techniques). See the :mod:`~pysatl_cpd.algorithms.online` docstring for a
  detailed overview, mathematical formulations, and extended examples.

.. raw:: html

    <h2>Public API</h2>

Bayesian algorithms:

- ``AbstractBayesian`` -- Abstract base class implementing the BOCPD
  message-passing loop.
- ``BayesianCPFType`` -- String enum for change-point score function selection
  (``MAX_RUN_LENGTH`` or ``DROP``).
- ``BayesianOnlineCPDConfiguration`` -- Base configuration dataclass shared by
  all Bayesian online detectors.
- ``BayesianOnlineCPDState`` -- Base state dataclass capturing the BOCPD
  run-length posterior snapshot.
- ``UnivariateGaussianConjugateBOCPD`` -- Concrete BOCPD detector for univariate
  data with a Normal-Inverse-Gamma conjugate prior.
- ``UnivariateGaussianConjugateBOCPDConfiguration`` -- Configuration dataclass
  for the univariate Gaussian conjugate detector.
- ``UnivariateGaussianConjugateBOCPDState`` -- State dataclass for the
  univariate Gaussian conjugate detector.

Control chart algorithms:

- ``ShewhartControlChart`` -- Shewhart control chart with a sliding-window
  statistic for mean-shift detection.
- ``ShewhartControlChartConfiguration`` -- Configuration dataclass for the
  Shewhart chart.
- ``ShewhartControlChartState`` -- State snapshot for the Shewhart chart.

CUSUM algorithms:

- ``PageTwoSidedCusum`` -- Two-sided Page CUSUM for Gaussian mean-shift
  detection (univariate and multivariate).
- ``PageTwoSidedCusumConfiguration`` -- Configuration dataclass for
  ``PageTwoSidedCusum``.
- ``PageTwoSidedCusumState`` -- State snapshot for ``PageTwoSidedCusum``.
- ``CrosierCusum`` -- Multivariate CUSUM based on the Crosier statistic with
  norm-based shrinkage.
- ``CrosierCusumConfiguration`` -- Configuration dataclass for ``CrosierCusum``.
- ``CrosierCusumState`` -- State snapshot for ``CrosierCusum``.
- ``VarianceTwoSidedCUSUM`` -- Two-sided CUSUM for univariate variance (scale)
  change detection.
- ``VarianceTwoSidedCusumConfiguration`` -- Configuration dataclass for
  ``VarianceTwoSidedCUSUM``.
- ``VarianceTwoSidedCusumState`` -- State snapshot for
  ``VarianceTwoSidedCUSUM``.
- ``AutoregressiveCUSUM`` -- CUSUM for univariate autoregressive Gaussian time
  series using Page's CPF on AR model residuals.
- ``AutoregressiveCusumConfiguration`` -- Configuration dataclass for
  ``AutoregressiveCUSUM``.
- ``AutoregressiveCusumState`` -- State snapshot for ``AutoregressiveCUSUM``.

Notes
-----
All algorithms require a ``learning_period_size`` of initial observations for
parameter estimation before producing non-zero detection statistics.

``AutoregressiveCUSUM`` and ``VarianceTwoSidedCUSUM`` accept only univariate
observations. ``CrosierCusum``, ``PageTwoSidedCusum``,
``ShewhartControlChart``, and ``UnivariateGaussianConjugateBOCPD`` support
multivariate input (though the BOCPD detector is designed for univariate data).

``AutoregressiveCUSUM`` requires the optional ``arch`` package. Install it via
``poetry add arch``.

For detailed mathematical formulations, component architecture, and extended
usage examples, consult the :mod:`~pysatl_cpd.algorithms.online` docstring and
the individual subpackage docstrings.

Examples
--------
Import and use a Shewhart control chart with ``OnlineResetDetector``:

>>> import numpy as np
>>> from pysatl_cpd.algorithms import ShewhartControlChart
>>> from pysatl_cpd.core.online import OnlineResetDetector
>>> from pysatl_cpd.data.generator import preset_dataset
>>> dataset = preset_dataset(
...     "mean_shifts", n_series=1, seed=7, series_length=180, n_features=1
... )
>>> provider = dataset[0]
>>> detector = OnlineResetDetector(
...     ShewhartControlChart(learning_period_size=30, window_size=10),
...     threshold=2.0,
... )
>>> trace = detector.detect(provider)
>>> len(trace.detected_change_points) > 0
True

Run a Bayesian BOCPD detector on a synthetic series:

>>> from pysatl_cpd.algorithms import UnivariateGaussianConjugateBOCPD
>>> algo = UnivariateGaussianConjugateBOCPD(
...     learning_period_size=20,
...     hazard_lambda=50.0,
... )
>>> rng = np.random.default_rng(42)
>>> series = np.concatenate([
...     rng.normal(0.0, 1.0, 30),
...     rng.normal(3.0, 1.0, 30),
... ])
>>> scores = [algo.process(x) for x in series]
>>> all(s == 0.0 for s in scores[:20])
True

Run a CUSUM detector and inspect its state:

>>> from pysatl_cpd.algorithms import PageTwoSidedCusum
>>> algo = PageTwoSidedCusum(learning_period_size=10, delta=0.5)
>>> for v in [0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.1, -0.3, 0.0, 0.1, 2.0, 2.5]:
...     _ = algo.process(np.array([v]))
>>> algo.state.is_in_learning_period
False
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online import *
from pysatl_cpd.algorithms.online import __all__ as _online_all

__all__ = [
    *_online_all,
]

del _online_all
