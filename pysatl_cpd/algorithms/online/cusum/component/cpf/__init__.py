# -*- coding: ascii -*-
"""
CUSUM change-point function (CPF) components.

This subpackage provides concrete implementations of the
:class:`~pysatl_cpd.algorithms.online.cusum.abstracts.changepoint_func.ICusumChangepointFunc`
protocol for use within the generalized CUSUM framework. Each CPF class
maintains an internal statistic that is updated incrementally from monitoring
observations and exposes a scalar ``value`` property used by the CUSUM
algorithm to decide whether a change-point has occurred.

The two implementations cover different detection scenarios:

- :class:`ChangepointFuncUnivariatePageCUSUM` accumulates positive and/or
  negative deviations from a reference value, supporting one-sided or
  two-sided detection on univariate (dim=1) inputs.
- :class:`ChangepointFuncCrosierCUSUM` applies norm-based shrinkage to a
  vector-valued accumulated statistic, making it suitable for multivariate
  monitoring spaces.

These components are not typically instantiated directly by end users.
Instead, they are wired into higher-level algorithm classes such as
:class:`~pysatl_cpd.algorithms.online.cusum.PageTwoSidedCusum` and
:class:`~pysatl_cpd.algorithms.online.cusum.CrosierCusum` as the
``changepoint_func`` argument of
:class:`~pysatl_cpd.algorithms.online.cusum.abstracts.generalized_cusum.GeneralizedCUSUM`.

.. raw:: html

    <h2>Public API</h2>

- ``ChangepointFuncUnivariatePageCUSUM`` -- Univariate Page CUSUM statistic
  with configurable detection side (``"pos"``, ``"neg"``, or ``"both"``).
- ``ChangepointFuncCrosierCUSUM`` -- Crosier-style vector CUSUM statistic
  with norm-based shrinkage controlled by a ``delta`` parameter.

Examples
--------
Create and update a two-sided Page CUSUM statistic:

>>> import numpy as np
>>> from pysatl_cpd.algorithms.online.cusum.component.cpf import (
...     ChangepointFuncUnivariatePageCUSUM,
... )
>>> cpf = ChangepointFuncUnivariatePageCUSUM(delta=0.5, side="both")
>>> cpf.update(np.array([1.2]))
>>> cpf.value
0.7
>>> cpf.update(np.array([-0.8]))
>>> cpf.reset()
>>> cpf.value
0.0

Create and update a Crosier CUSUM statistic for 3-dimensional observations:

>>> from pysatl_cpd.algorithms.online.cusum.component.cpf import (
...     ChangepointFuncCrosierCUSUM,
... )
>>> cpf = ChangepointFuncCrosierCUSUM(dim=3, delta=1.0)
>>> cpf.update(np.array([1.0, 2.0, 0.5]))
>>> cpf.value > 0.0
True
>>> cpf.reset()
>>> cpf.value
0.0

Notes
-----
- Both classes implement the ``ICusumChangepointFunc`` protocol defined in
  the ``abstracts`` subpackage. See that module for the full interface
  contract.
- ``ChangepointFuncUnivariatePageCUSUM`` requires observations with
  ``shape[0] == 1``; passing a multidimensional array raises ``ValueError``.
- ``ChangepointFuncCrosierCUSUM`` infers the observation dimensionality from
  the first call to ``update()``; the ``dim`` constructor argument is
  retained for API compatibility but does not pre-allocate the statistic.
- This subpackage depends on NumPy.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.cusum.component.cpf.crosier import ChangepointFuncCrosierCUSUM
from pysatl_cpd.algorithms.online.cusum.component.cpf.page import ChangepointFuncUnivariatePageCUSUM

__all__ = [
    "ChangepointFuncUnivariatePageCUSUM",
    "ChangepointFuncCrosierCUSUM",
]
