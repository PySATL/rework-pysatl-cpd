# -*- coding: ascii -*-
"""
Monitoring schemas for the generalized CUSUM algorithm.

This package provides concrete implementations of the
:class:`~pysatl_cpd.algorithms.online.cusum.abstracts.monitoring.IMonitoringSchema`
interface. Each monitoring schema transforms raw observations and estimated
model parameters into a monitoring-space residual that the generalized CUSUM
accumulates to detect change points.

Three monitoring strategies are available:

- ``GaussianMonitoringSchema`` whitens multivariate observation residuals
  using the inverse square root of the estimated covariance matrix. Suitable
  for detecting mean shifts in multivariate Gaussian data.
- ``GaussianARMonitoringSchema`` computes standardised one-step-ahead
  forecast residuals from an autoregressive model. Designed for detecting
  structural breaks in univariate time series with temporal dependence.
- ``VarianceMonitoringSchema`` converts consecutive-observation differences
  into approximately standardised variance-change statistics. Targets
  detection of volatility shifts in univariate data.

.. raw:: html

    <h2>Public API</h2>

- **GaussianARMonitoringSchema** -- Standardised AR forecast residual
  monitoring. Works with
  :class:`~pysatl_cpd.algorithms.online.cusum.component.estimator.gaussian_ar.EstimatesGaussianAR`.
- **GaussianMonitoringSchema** -- Covariance-whitened residual monitoring.
  Works with :class:`~pysatl_cpd.algorithms.online.cusum.component.estimator.gaussian_mle.EstimatesGaussianMLE`.
- **VarianceMonitoringSchema** -- Variance-change residual monitoring.
  Works with :class:`~pysatl_cpd.algorithms.online.cusum.component.estimator.gaussian_mle.EstimatesGaussianMLE`.

.. raw:: html

    <h2>Examples</h2>

Gaussian monitoring for multivariate mean-shift detection::

    >>> import numpy as np
    >>> from pysatl_cpd.algorithms.online.cusum.component.monitoring import (
    ...     GaussianMonitoringSchema,
    ... )
    >>> from pysatl_cpd.algorithms.online.cusum.component.estimator import (
    ...     GaussianMLESchema,
    ... )
    >>> schema = GaussianMonitoringSchema(cov_reg=1e-6)
    >>> estimator = GaussianMLESchema(adaptive=True)
    >>> train_data = [np.array([0.0, 1.0]) for _ in range(30)]
    >>> estimator.train(train_data)
    >>> obs = np.array([0.5, 1.2])
    >>> residual = schema.evaluate(obs, estimator.estimates)
    >>> residual.shape
    (2,)

Variance monitoring for univariate volatility detection::

    >>> import numpy as np
    >>> from pysatl_cpd.algorithms.online.cusum.component.monitoring import (
    ...     VarianceMonitoringSchema,
    ... )
    >>> from pysatl_cpd.algorithms.online.cusum.component.estimator import (
    ...     GaussianMLESchema,
    ... )
    >>> schema = VarianceMonitoringSchema()
    >>> estimator = GaussianMLESchema(adaptive=True)
    >>> train_data = [np.array([x]) for x in np.random.default_rng(0).standard_normal(30)]
    >>> estimator.train(train_data)
    >>> obs = np.array([0.5])
    >>> residual = schema.evaluate(obs, estimator.estimates)
    >>> residual.shape
    (1,)

Autoregressive monitoring for univariate time-series break detection::

    >>> import numpy as np
    >>> from pysatl_cpd.algorithms.online.cusum.component.monitoring import (
    ...     GaussianARMonitoringSchema,
    ... )
    >>> from pysatl_cpd.algorithms.online.cusum.component.estimator import (
    ...     GaussianARSchema,
    ... )
    >>> schema = GaussianARMonitoringSchema()
    >>> estimator = GaussianARSchema(autoreg_order=2, adaptive=True)
    >>> rng = np.random.default_rng(0)
    >>> train_data = [np.array([x]) for x in rng.standard_normal(50)]
    >>> estimator.train(train_data)
    >>> obs = np.array([0.5])
    >>> residual = schema.evaluate(obs, estimator.estimates)
    >>> residual.shape
    (1,)

.. raw:: html

    <h2>Notes</h2>

Notes
-----
All monitoring schemas implement the
:class:`~pysatl_cpd.algorithms.online.cusum.abstracts.monitoring.IMonitoringSchema`
protocol and are consumed internally by the CUSUM algorithm implementations
in ``pysatl_cpd.algorithms.online.cusum.algorithm``. End users typically
interact with these schemas indirectly through the high-level CUSUM classes
such as :class:`~pysatl_cpd.algorithms.online.cusum.PageTwoSidedCusum`,
:class:`~pysatl_cpd.algorithms.online.cusum.CrosierCusum`,
:class:`~pysatl_cpd.algorithms.online.cusum.VarianceTwoSidedCUSUM`, and
:class:`~pysatl_cpd.algorithms.online.cusum.AutoregressiveCUSUM`.

``GaussianARMonitoringSchema`` requires the optional ``arch`` package.
Install it via ``poetry add arch`` before use.

``GaussianMonitoringSchema`` supports multivariate observations of any
dimensionality. ``GaussianARMonitoringSchema`` and ``VarianceMonitoringSchema``
are restricted to univariate (dim=1) observations.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.cusum.component.monitoring.gaussian import GaussianMonitoringSchema
from pysatl_cpd.algorithms.online.cusum.component.monitoring.gaussian_arm import GaussianARMonitoringSchema
from pysatl_cpd.algorithms.online.cusum.component.monitoring.variance import VarianceMonitoringSchema

__all__ = [
    "GaussianARMonitoringSchema",
    "GaussianMonitoringSchema",
    "VarianceMonitoringSchema",
]
