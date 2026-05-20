# -*- coding: ascii -*-
"""
CUSUM algorithm components.

Provides the building blocks for the generalized CUSUM framework: change-point
functions (CPF), parameter estimators, and monitoring schemas. These components
are wired together by higher-level algorithm classes such as
:class:`~pysatl_cpd.algorithms.online.cusum.PageTwoSidedCusum`,
:class:`~pysatl_cpd.algorithms.online.cusum.CrosierCusum`,
:class:`~pysatl_cpd.algorithms.online.cusum.VarianceTwoSidedCUSUM`, and
:class:`~pysatl_cpd.algorithms.online.cusum.AutoregressiveCUSUM`.

.. raw:: html

    <h2>Public API</h2>

- ``ChangepointFuncCrosierCUSUM`` -- Crosier-style vector CUSUM statistic
  with norm-based shrinkage for multivariate monitoring.
- ``ChangepointFuncUnivariatePageCUSUM`` -- Univariate Page CUSUM statistic
  supporting one-sided or two-sided detection.
- ``GaussianARSchema`` -- Univariate autoregressive parameter estimator
  backed by the ``arch`` package.
- ``GaussianARMonitoringSchema`` -- Standardised AR forecast residual
  monitoring for univariate time-series break detection.
- ``GaussianMLESchema`` -- Gaussian mean and covariance estimator using
  Welford's online algorithm; supports multivariate observations.
- ``GaussianMonitoringSchema`` -- Covariance-whitened residual monitoring
  for multivariate mean-shift detection.
- ``VarianceMonitoringSchema`` -- Variance-change residual monitoring for
  univariate volatility shift detection.

.. raw:: html

    <h2>Subpackages</h2>

- ``cpf`` -- Change-point function implementations. See the
  :mod:`~pysatl_cpd.algorithms.online.cusum.component.cpf` docstring for
  details.
- ``estimator`` -- Online parameter estimation schemas. See the
  :mod:`~pysatl_cpd.algorithms.online.cusum.component.estimator` docstring
  for details.
- ``monitoring`` -- Monitoring/residual transformation schemas. See the
  :mod:`~pysatl_cpd.algorithms.online.cusum.component.monitoring` docstring
  for details.

Examples
--------
Compose a two-sided Page CUSUM from its components:

>>> import numpy as np
>>> from pysatl_cpd.algorithms.online.cusum.component import (
...     ChangepointFuncUnivariatePageCUSUM,
...     GaussianMLESchema,
...     GaussianMonitoringSchema,
... )
>>> cpf = ChangepointFuncUnivariatePageCUSUM(delta=0.5, side="both")
>>> estimator = GaussianMLESchema(adaptive=True)
>>> train_data = [np.array([x]) for x in [0.0] * 30]
>>> estimator.train(train_data)
>>> monitor = GaussianMonitoringSchema(cov_reg=1e-6)
>>> obs = np.array([3.0])
>>> residual = monitor.evaluate(obs, estimator.estimates)
>>> cpf.update(residual)
>>> cpf.value > 0.0
True

Notes
-----
- ``GaussianARSchema`` and ``GaussianARMonitoringSchema`` require the
  optional ``arch`` dependency. Install it via ``poetry add arch``.
- ``GaussianMLESchema`` and ``GaussianMonitoringSchema`` support
  multivariate observations of any dimension.
- ``ChangepointFuncUnivariatePageCUSUM``, ``GaussianARSchema``,
  ``GaussianARMonitoringSchema``, and ``VarianceMonitoringSchema`` are
  restricted to univariate (dim=1) observations.
- All components depend on NumPy.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.cusum.component.cpf import (
    ChangepointFuncCrosierCUSUM,
    ChangepointFuncUnivariatePageCUSUM,
)
from pysatl_cpd.algorithms.online.cusum.component.estimator import GaussianARSchema, GaussianMLESchema
from pysatl_cpd.algorithms.online.cusum.component.monitoring import (
    GaussianARMonitoringSchema,
    GaussianMonitoringSchema,
    VarianceMonitoringSchema,
)

__all__ = [
    "ChangepointFuncCrosierCUSUM",
    "ChangepointFuncUnivariatePageCUSUM",
    "GaussianARSchema",
    "GaussianARMonitoringSchema",
    "GaussianMLESchema",
    "GaussianMonitoringSchema",
    "VarianceMonitoringSchema",
]
