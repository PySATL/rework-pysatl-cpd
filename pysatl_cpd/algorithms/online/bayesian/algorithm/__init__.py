# -*- coding: ascii -*-
"""Bayesian online change-point detection algorithm implementations.

This module provides the core algorithm classes for Bayesian online change-point
detection (BOCPD) following the Adams and MacKay (2007) message-passing framework.
It exposes an abstract base class that implements the BOCPD update loop and a
concrete univariate Gaussian conjugate detector ready for use.

.. raw:: html

    <h2>Public API</h2>

Classes
~~~~~~~

AbstractBayesian
    Abstract base class implementing the BOCPD message-passing loop. Subclasses
    must supply a hazard function, likelihood model, and change-point score
    function (CPF). See ``abstract_bayesian.py`` for details.

BayesianOnlineCPDConfiguration
    Frozen dataclass holding base configuration fields shared by all Bayesian
    online detectors (learning period, window size, CPF type).

BayesianOnlineCPDState
    Frozen dataclass capturing a snapshot of the base algorithm state, including
    the current time step and run-length log-posterior distribution.

UnivariateGaussianConjugateBOCPD
    Concrete BOCPD detector for univariate data with a Normal-Inverse-Gamma
    conjugate prior, constant hazard model, and selectable CPF strategy.
    This is the primary ready-to-use class in this module.

UnivariateGaussianConjugateBOCPDConfiguration
    Configuration dataclass extending the base with hazard and prior hyper-
    parameters (hazard_lambda, prior_mu, prior_k, prior_alpha, prior_beta).

UnivariateGaussianConjugateBOCPDState
    State dataclass extending the base with per-run-length posterior parameters
    (mu, k, alpha, beta arrays).

Notes
-----
- All algorithms implement the ``OnlineAlgorithm`` interface from
  ``pysatl_cpd.core.online`` and can be wrapped by ``OnlineResetDetector``
  for threshold-based detection with reset semantics.
- Scores are zero during the learning period. After that, the CPF component
  determines the detection statistic (max run-length probability or drop
  probability, controlled by ``BayesianCPFType``).
- The ``window`` parameter truncates the run-length posterior to limit memory
  and computation for long streams.
- Dependencies: ``numpy``, ``scipy.special.logsumexp``.

Examples
--------
Create a univariate Gaussian conjugate BOCPD detector and process observations
one at a time:

>>> import numpy as np
>>> from pysatl_cpd.algorithms.online.bayesian.algorithm import (
...     UnivariateGaussianConjugateBOCPD,
...     UnivariateGaussianConjugateBOCPDConfiguration,
... )
>>> from pysatl_cpd.algorithms.online.bayesian._enum import BayesianCPFType
>>>
>>> algo = UnivariateGaussianConjugateBOCPD(
...     learning_period_size=20,
...     hazard_lambda=50.0,
...     prior_mu=0.0,
...     prior_k=1.0,
...     prior_alpha=1.0,
...     prior_beta=1.0,
...     cpf_type=BayesianCPFType.MAX_RUN_LENGTH,
... )
>>> algo.name
'UnivariateGaussianConjugateBOCPD'
>>> algo.configuration  # doctest: +ELLIPSIS
UnivariateGaussianConjugateBOCPDConfiguration(learning_period_size=20, ...

Process a short synthetic series with a mean shift:

>>> rng = np.random.default_rng(42)
>>> series = np.concatenate([rng.normal(0.0, 1.0, 30), rng.normal(3.0, 1.0, 30)])
>>> scores = [algo.process(x) for x in series]
>>> # Scores are 0 during learning period
>>> all(s == 0.0 for s in scores[:20])
True
>>> # Scores become positive after learning period
>>> any(s > 0.0 for s in scores[20:])
True

Access the current state snapshot:

>>> state = algo.state
>>> state.t
60
>>> len(state.run_length_log_posterior)  # doctest: +ELLIPSIS
60...

Reset the algorithm to its initial state:

>>> algo.reset()
>>> algo.t
0

Create an independent copy with identical configuration:

>>> clone = algo.recreate()
>>> clone.configuration == algo.configuration
True

Use with ``OnlineResetDetector`` for threshold-based detection:

>>> from pysatl_cpd.core.online import OnlineResetDetector
>>> from pysatl_cpd.data.generator import preset_dataset
>>>
>>> dataset = preset_dataset("mean_shifts", n_series=1, seed=7, series_length=180, n_features=1)
>>> provider = dataset[0]
>>> detector = OnlineResetDetector(
...     UnivariateGaussianConjugateBOCPD(learning_period_size=30, hazard_lambda=50.0),
...     threshold=0.5,
... )
>>> trace = detector.detect(provider)
>>> trace.detected_change_points  # doctest: +ELLIPSIS
[...]

Configure via the configuration dataclass directly:

>>> config = UnivariateGaussianConjugateBOCPDConfiguration(
...     learning_period_size=10,
...     hazard_lambda=25.0,
...     prior_mu=0.0,
...     prior_k=2.0,
...     prior_alpha=1.0,
...     prior_beta=1.0,
... )
>>> config.window is None
True
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.bayesian.algorithm.abstract_bayesian import (
    AbstractBayesian,
    BayesianOnlineCPDConfiguration,
    BayesianOnlineCPDState,
)
from pysatl_cpd.algorithms.online.bayesian.algorithm.univariate_gaussian_conjugate import (
    UnivariateGaussianConjugateBOCPD,
    UnivariateGaussianConjugateBOCPDConfiguration,
    UnivariateGaussianConjugateBOCPDState,
)

__all__ = [
    "AbstractBayesian",
    "BayesianOnlineCPDConfiguration",
    "BayesianOnlineCPDState",
    "UnivariateGaussianConjugateBOCPD",
    "UnivariateGaussianConjugateBOCPDConfiguration",
    "UnivariateGaussianConjugateBOCPDState",
]
