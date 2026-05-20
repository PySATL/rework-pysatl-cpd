# -*- coding: ascii -*-
"""
CUSUM estimator components.

Provides estimating schemas that maintain online parameter estimates for
generalized CUSUM algorithms. Each schema implements the
:class:`~pysatl_cpd.algorithms.online.cusum.abstracts.estimator.IEstimatingSchema`
interface, supporting training on a learning sample, optional adaptive
updates with new observations, and state reset.

Public API
----------

- **GaussianMLESchema**: Estimates mean and covariance using Welford's
  numerically stable online algorithm. Supports multivariate observations.
  See ``gaussian_mle`` for details.
- **GaussianARSchema**: Fits a univariate autoregressive (AR) model via the
  ``arch`` package. Requires the optional ``arch`` dependency.
  See ``gaussian_ar`` for details.

Examples
--------

Gaussian mean/covariance estimation:

>>> import numpy as np
>>> from pysatl_cpd.algorithms.online.cusum.component.estimator import GaussianMLESchema
>>> schema = GaussianMLESchema(adaptive=True)
>>> schema.train([np.array([1.0]), np.array([2.0]), np.array([3.0])])
>>> round(float(schema.mean[0]), 2)
2.0
>>> schema.update(np.array([4.0]))
>>> round(float(schema.mean[0]), 2)
2.5

Autoregressive estimation (requires the ``arch`` package):

>>> from pysatl_cpd.algorithms.online.cusum.component.estimator import GaussianARSchema
>>> schema = GaussianARSchema(autoreg_order=1, adaptive=True)
>>> schema.train([np.array([1.0]), np.array([2.0]), np.array([3.0]), np.array([4.0])])
>>> estimates = schema.estimates
>>> round(estimates["noise_variance"], 4)
0.0

Notes
-----

- ``GaussianARSchema`` requires the optional ``arch`` dependency. Install it
  via ``poetry add arch`` or your preferred method.
- ``GaussianARSchema`` only supports univariate observations (dim=1).
- ``GaussianMLESchema`` supports multivariate observations of any dimension.
- Both schemas implement the ``IEstimatingSchema`` protocol defined in
  ``pysatl_cpd.algorithms.online.cusum.abstracts.estimator``.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.cusum.component.estimator.gaussian_ar import GaussianARSchema
from pysatl_cpd.algorithms.online.cusum.component.estimator.gaussian_mle import GaussianMLESchema

__all__ = ["GaussianARSchema", "GaussianMLESchema"]
