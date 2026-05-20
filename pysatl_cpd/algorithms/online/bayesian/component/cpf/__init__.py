# -*- coding: ascii -*-
"""Bayesian change-point score functions (CPF).

This subpackage provides concrete implementations of the ``IBayesianCPF``
protocol for converting run-length log-posterior arrays into scalar
change-point scores during online Bayesian change-point detection.

.. raw:: html

    <h2>Public API</h2>

- **DropCPF**: Computes the positive drop in maximal-run-length probability
  between consecutive steps. Useful for detecting sudden regime changes by
  tracking decreases in the probability of the current run continuing.
- **MaxRunLengthCPF**: Returns one minus the probability of the maximal run
  length state (the last element of the posterior). Provides a simple
  instantaneous score based on how unlikely the longest run is.

Both classes implement ``calculate()`` for scoring and ``clear()`` for state
reset. See the ``IBayesianCPF`` protocol in
``pysatl_cpd.algorithms.online.bayesian.protocol.cpf`` for the full interface.

.. raw:: html

    <h2>Examples</h2>

MaxRunLengthCPF usage::

    >>> import numpy as np
    >>> from pysatl_cpd.algorithms.online.bayesian.component.cpf import MaxRunLengthCPF
    >>> cpf = MaxRunLengthCPF()
    >>> # Run-length log-posterior: log([0.2, 0.8])
    >>> log_posterior = np.log(np.array([0.2, 0.8], dtype=np.float64))
    >>> round(cpf.calculate(log_posterior), 4)
    0.2

DropCPF usage::

    >>> import numpy as np
    >>> from pysatl_cpd.algorithms.online.bayesian.component.cpf import DropCPF
    >>> cpf = DropCPF()
    >>> # First call always returns 0.0 (no previous state)
    >>> log_posterior_1 = np.log(np.array([0.2, 0.8], dtype=np.float64))
    >>> cpf.calculate(log_posterior_1)
    0.0
    >>> # Second call: probability of max run dropped from 0.8 to 0.5
    >>> log_posterior_2 = np.log(np.array([0.5, 0.5], dtype=np.float64))
    >>> round(cpf.calculate(log_posterior_2), 4)
    0.3
    >>> cpf.clear()
    >>> # After clear, next call returns 0.0 again
    >>> cpf.calculate(log_posterior_2)
    0.0

.. raw:: html

    <h2>Notes</h2>

- CPF instances are stateless except for ``DropCPF``, which tracks the
  previous step's max-run log-probability. Call ``clear()`` to reset.
- Empty input arrays always produce a score of ``0.0``.
- These components are typically constructed via the factory function
  ``get_cpf_function()`` in ``pysatl_cpd.algorithms.online.bayesian.utils``
  using the ``BayesianCPFType`` enum.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.bayesian.component.cpf.drop import DropCPF
from pysatl_cpd.algorithms.online.bayesian.component.cpf.max_run_length import MaxRunLengthCPF

__all__ = ["DropCPF", "MaxRunLengthCPF"]
