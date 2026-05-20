# -*- coding: ascii -*-
"""Likelihood models for Bayesian online change-point detection.

This package provides likelihood components used in Bayesian online
change-point detection (BOCPD) algorithms. Likelihood models compute
predictive probabilities for new observations under both prior and
posterior distributions, which are essential for updating run-length
distributions in the BOCPD framework.

The ``gaussian_conjugate`` module implements a Normal-Inverse-Gamma
conjugate likelihood with Student-t predictive density. See its
module-level docstring for details.

.. raw:: html

    <h2>Public API</h2>

- **GaussianConjugate**: Normal-Inverse-Gamma conjugate likelihood that
  maintains posterior sufficient statistics and computes Student-t
  predictive log-probabilities. Implements the ``ILikelihood`` protocol.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Create a Gaussian conjugate likelihood and compute predictive scores::

    >>> import numpy as np
    >>> from pysatl_cpd.algorithms.online.bayesian.component.likelihood import GaussianConjugate
    >>> likelihood = GaussianConjugate(mu_0=0.0, k_0=1.0, alpha_0=1.0, beta_0=1.0)
    >>> prior_loglik = likelihood.predict(np.float64(0.5))
    >>> prior_loglik.shape
    (1,)

Update the posterior with observations and compute posterior predictive scores::

    >>> likelihood.update(np.float64(0.5))
    >>> post_loglik = likelihood.predict(np.float64(0.5))
    >>> post_loglik.shape
    (2,)

Reset the likelihood state back to the prior::

    >>> likelihood.clear()
    >>> likelihood.predict(np.float64(0.5)).shape
    (1,)

Notes
-----
The ``GaussianConjugate`` class uses a Normal-Inverse-Gamma prior, which
yields a Student-t predictive distribution. Hyperparameters ``k_0``,
``alpha_0``, and ``beta_0`` must all be strictly positive.

The ``predict`` method returns an array whose first element is the prior
predictive log-likelihood and remaining elements are posterior predictive
log-likelihoods for accumulated run-length states. The optional ``window``
parameter limits the number of posterior states considered.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.bayesian.component.likelihood.gaussian_conjugate import GaussianConjugate

__all__ = ["GaussianConjugate"]
