# -*- coding: ascii -*-
"""Protocols for Bayesian online components.

This module defines structural typing interfaces (PEP 544 protocols) used by
Bayesian online change-point detection algorithms. The three protocols --
``IBayesianCPF``, ``IHazard``, and ``ILikelihood`` -- describe the contracts
that concrete component implementations must satisfy.

.. raw:: html

    <h2>Public API</h2>

- ``IBayesianCPF`` -- Protocol for scalar change-point score functions.
  Converts a run-length log-posterior vector into a single detection score.
  See ``cpf.py`` for method details.
- ``IHazard`` -- Protocol for hazard models. Returns log-hazard and
  log-survival arrays given run-length indices. See ``hazard.py`` for
  method details.
- ``ILikelihood`` -- Protocol for predictive likelihood models. Provides
  ``predict`` and ``update`` methods for sequential Bayesian inference.
  See ``likelihood.py`` for method details.

.. raw:: html

    <h2>Examples</h2>

Create concrete instances that satisfy each protocol::

    >>> import numpy as np
    >>> from pysatl_cpd.algorithms.online.bayesian.component.cpf import DropCPF
    >>> from pysatl_cpd.algorithms.online.bayesian.component.hazard import ConstantHazard
    >>> from pysatl_cpd.algorithms.online.bayesian.component.likelihood import GaussianConjugate
    >>> from pysatl_cpd.algorithms.online.bayesian.protocol import IBayesianCPF, IHazard, ILikelihood
    >>> cpf: IBayesianCPF = DropCPF()
    >>> hazard: IHazard = ConstantHazard(lambda_=100.0)
    >>> likelihood: ILikelihood = GaussianConjugate(mu_0=0.0, k_0=1.0, alpha_0=1.0, beta_0=1.0)

Use the likelihood protocol to process observations sequentially::

    >>> likelihood.clear()
    >>> log_probs = likelihood.predict(np.float64(1.5))
    >>> likelihood.update(np.float64(1.5))

Use the hazard protocol to compute hazard values::

    >>> run_lengths = np.arange(5)
    >>> log_h, log_s = hazard.hazard(run_lengths)

Use the CPF protocol to compute a change-point score::

    >>> log_posterior = np.array([-10.0, -5.0, -2.0, -1.0, 0.0])
    >>> score = cpf.calculate(log_posterior)
    >>> cpf.clear()

.. raw:: html

    <h2>Notes</h2>

- All protocols use structural subtyping (``typing.Protocol``). Any class
  that implements the required methods is compatible -- no explicit
  inheritance is needed.
- These protocols are consumed by the factory helpers in
  ``pysatl_cpd.algorithms.online.bayesian.utils`` and by the concrete
  component implementations in
  ``pysatl_cpd.algorithms.online.bayesian.component``.
- All log-space computations use ``numpy.float64``. Arrays use
  ``numpy.typing.NDArray`` with appropriate dtypes.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.bayesian.protocol.cpf import IBayesianCPF
from pysatl_cpd.algorithms.online.bayesian.protocol.hazard import IHazard
from pysatl_cpd.algorithms.online.bayesian.protocol.likelihood import ILikelihood

__all__ = ["IBayesianCPF", "IHazard", "ILikelihood"]
