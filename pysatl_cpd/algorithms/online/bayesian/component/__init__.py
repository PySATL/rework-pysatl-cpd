# -*- coding: ascii -*-
"""Components for Bayesian online change-point detection.

This package provides the three core building blocks of the Bayesian online
change-point detection (BOCPD) framework: hazard models, likelihood models,
and change-point score functions (CPF). Each component implements a protocol
defined in the sibling ``protocol`` subpackage and can be composed together
to configure a full BOCPD algorithm.

.. raw:: html

    <h2>Public API</h2>

- **ConstantHazard**: Constant hazard model with a fixed expected run length
  timescale. Implements the ``IHazard`` protocol.
- **GaussianConjugate**: Normal-Inverse-Gamma conjugate likelihood with
  Student-t predictive density. Implements the ``ILikelihood`` protocol.
- **MaxRunLengthCPF**: Returns one minus the probability of the maximal run
  length state. Implements the ``IBayesianCPF`` protocol.
- **DropCPF**: Computes the positive drop in maximal-run-length probability
  between consecutive steps. Implements the ``IBayesianCPF`` protocol.

.. raw:: html

    <h2>Subpackages</h2>

- ``cpf`` -- change-point score function implementations (``DropCPF``,
  ``MaxRunLengthCPF``). See its module docstring for details.
- ``hazard`` -- hazard model implementations (``ConstantHazard``). See its
  module docstring for details.
- ``likelihood`` -- likelihood model implementations (``GaussianConjugate``).
  See its module docstring for details.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Construct all three component types and use them together in a BOCPD-style
setup::

    >>> import numpy as np
    >>> from pysatl_cpd.algorithms.online.bayesian.component import (
    ...     ConstantHazard,
    ...     DropCPF,
    ...     GaussianConjugate,
    ...     MaxRunLengthCPF,
    ... )
    >>> hazard = ConstantHazard(lambda_=100.0)
    >>> likelihood = GaussianConjugate(mu_0=0.0, k_0=1.0, alpha_0=1.0, beta_0=1.0)
    >>> cpf_max = MaxRunLengthCPF()
    >>> cpf_drop = DropCPF()

Compute predictive log-probabilities and update the likelihood posterior::

    >>> x = np.float64(0.5)
    >>> prior_scores = likelihood.predict(x)
    >>> prior_scores.shape
    (1,)
    >>> likelihood.update(x)
    >>> post_scores = likelihood.predict(x)
    >>> post_scores.shape
    (2,)

Compute hazard and survival values for a set of run lengths::

    >>> run_lengths = np.array([0, 1, 2], dtype=np.intp)
    >>> log_h, log_surv = hazard.hazard(run_lengths)
    >>> log_h.shape
    (3,)

Score a run-length log-posterior with both CPF variants::

    >>> log_posterior = np.log(np.array([0.3, 0.7], dtype=np.float64))
    >>> round(cpf_max.calculate(log_posterior), 4)
    0.3
    >>> cpf_drop.calculate(log_posterior)
    0.0

Notes
-----
Components are typically constructed via the factory functions in
``pysatl_cpd.algorithms.online.bayesian.utils`` rather than imported
directly. See ``get_hazard_function()``, ``get_likelihood_function()``,
and ``get_cpf_function()``.

All hazard and likelihood computations use log-space values for numerical
stability. The ``GaussianConjugate`` hyperparameters ``k_0``, ``alpha_0``,
and ``beta_0`` must all be strictly positive.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.bayesian.component.cpf import DropCPF, MaxRunLengthCPF
from pysatl_cpd.algorithms.online.bayesian.component.hazard import ConstantHazard
from pysatl_cpd.algorithms.online.bayesian.component.likelihood import GaussianConjugate

__all__ = ["ConstantHazard", "DropCPF", "GaussianConjugate", "MaxRunLengthCPF"]
