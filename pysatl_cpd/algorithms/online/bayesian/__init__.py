# -*- coding: ascii -*-
"""Bayesian online change-point detection algorithms.

This package implements Bayesian online change-point detection (BOCPD) following
the Adams and MacKay (2007) message-passing framework. It provides an abstract
base class that implements the BOCPD update loop, a ready-to-use univariate
Gaussian conjugate detector, factory functions for constructing components, and
an enum for selecting change-point score strategies.

The package is organized into four subpackages:

- ``algorithm`` -- core algorithm implementations including the abstract base
  class and the univariate Gaussian conjugate detector. See its docstring for
  details.
- ``component`` -- building block components (hazard models, likelihood models,
  and change-point score functions). See its docstring for details.
- ``protocol`` -- structural typing interfaces (PEP 544 protocols) that define
  the contracts for components. See its docstring for details.
- ``utils`` -- factory functions for constructing configured component instances.
  See its docstring for details.

.. raw:: html

    <h2>Public API</h2>

Classes
~~~~~~~

AbstractBayesian
    Abstract base class implementing the BOCPD message-passing loop. Subclasses
    must supply a hazard function, likelihood model, and change-point score
    function (CPF).

BayesianOnlineCPDConfiguration
    Frozen dataclass holding base configuration fields shared by all Bayesian
    online detectors (learning period, window size, CPF type).

BayesianOnlineCPDState
    Frozen dataclass capturing a snapshot of the base algorithm state, including
    the current time step and run-length log-posterior distribution.

UnivariateGaussianConjugateBOCPD
    Concrete BOCPD detector for univariate data with a Normal-Inverse-Gamma
    conjugate prior, constant hazard model, and selectable CPF strategy.
    This is the primary ready-to-use class in this package.

UnivariateGaussianConjugateBOCPDConfiguration
    Configuration dataclass extending the base with hazard and prior hyper-
    parameters (hazard_lambda, prior_mu, prior_k, prior_alpha, prior_beta).

UnivariateGaussianConjugateBOCPDState
    State dataclass extending the base with per-run-length posterior parameters
    (mu, k, alpha, beta arrays).

BayesianCPFType
    String enum selecting the change-point score function variant:
    ``MAX_RUN_LENGTH`` or ``DROP``.

Functions
~~~~~~~~~

get_hazard_function(lambda\_) -> IHazard
    Build a constant hazard model with the given expected run length.

get_likelihood_function(mu_0, k_0, alpha_0, beta_0) -> ILikelihood
    Build a Gaussian conjugate likelihood with the specified prior parameters.

get_cpf_function(cpf_type) -> IBayesianCPF
    Build a change-point score function selected by ``BayesianCPFType``.

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
>>> from pysatl_cpd.algorithms.online.bayesian import (
...     UnivariateGaussianConjugateBOCPD,
...     UnivariateGaussianConjugateBOCPDConfiguration,
...     BayesianCPFType,
... )
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
>>> all(s == 0.0 for s in scores[:20])
True
>>> any(s > 0.0 for s in scores[20:])
True

Access the current state snapshot:

>>> state = algo.state
>>> state.t
60
>>> len(state.run_length_log_posterior)  # doctest: +ELLIPSIS
60...

Reset the algorithm and create an independent copy:

>>> algo.reset()
>>> algo.t
0
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

Use factory functions to construct components:

>>> from pysatl_cpd.algorithms.online import BayesianCPFType
>>> from pysatl_cpd.algorithms.online.bayesian.utils import (
...     get_cpf_function,
...     get_hazard_function,
...     get_likelihood_function,
... )
>>> hazard = get_hazard_function(lambda_=10.0)
>>> likelihood = get_likelihood_function(mu_0=0.0, k_0=1.0, alpha_0=1.0, beta_0=1.0)
>>> cpf = get_cpf_function(BayesianCPFType.MAX_RUN_LENGTH)
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.bayesian._enum import BayesianCPFType
from pysatl_cpd.algorithms.online.bayesian.algorithm import (
    AbstractBayesian,
    BayesianOnlineCPDConfiguration,
    BayesianOnlineCPDState,
    UnivariateGaussianConjugateBOCPD,
    UnivariateGaussianConjugateBOCPDConfiguration,
    UnivariateGaussianConjugateBOCPDState,
)
from pysatl_cpd.algorithms.online.bayesian.utils import (
    get_cpf_function,
    get_hazard_function,
    get_likelihood_function,
)

__all__ = [
    "AbstractBayesian",
    "BayesianCPFType",
    "BayesianOnlineCPDConfiguration",
    "BayesianOnlineCPDState",
    "UnivariateGaussianConjugateBOCPD",
    "UnivariateGaussianConjugateBOCPDConfiguration",
    "UnivariateGaussianConjugateBOCPDState",
    "get_cpf_function",
    "get_hazard_function",
    "get_likelihood_function",
]
