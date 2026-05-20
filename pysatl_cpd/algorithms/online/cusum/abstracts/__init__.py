# -*- coding: ascii -*-
"""
Abstract base classes and protocols for generalized CUSUM detectors.

This subpackage defines the component interfaces and base class that concrete
CUSUM algorithms compose to form online change-point detectors. Every CUSUM
detector is built from three interchangeable parts:

- An **estimating schema** (``IEstimatingSchema``) that learns model
  parameters from a training sample and optionally adapts them online.
- A **monitoring schema** (``IMonitoringSchema``) that transforms raw
  observations and current parameter estimates into monitoring-space
  residuals.
- A **change-point function** (``ICusumChangepointFunc``) that accumulates
  monitoring residuals into a scalar detection statistic.

The ``GeneralizedCUSUM`` base class wires these components together, handles
the learning-period lifecycle, and implements the ``OnlineAlgorithm``
contract expected by the online solver and detector infrastructure.

.. raw:: html

    <h2>Public API</h2>

- ``GeneralizedCUSUM`` -- Abstract base class for configurable online CUSUM
  detectors. Combines an estimating schema, monitoring schema, and
  change-point function into a single ``OnlineAlgorithm``.
- ``GeneralizedCUSUMConfiguration`` -- Frozen configuration dataclass carrying
  ``learning_period_size`` and ``adaptive_estimation`` flags.
- ``GeneralizedCUSUMState`` -- Frozen state snapshot dataclass carrying the
  learning-period flag and current schema estimates.
- ``IEstimatingSchema`` -- Protocol for parameter-estimation components with
  ``train``, ``update``, ``reset``, and ``estimates`` members.
- ``IMonitoringSchema`` -- Protocol for observation-to-residual transforms
  with ``evaluate`` and ``reset`` members.
- ``ICusumChangepointFunc`` -- Protocol for scalar statistic accumulators
  with ``update``, ``value``, and ``reset`` members.
- ``ISchemaEstimates`` -- Base ``TypedDict`` for schema-estimate payloads.

.. raw:: html

    <h2>Submodules</h2>

- ``changepoint_func`` -- Defines ``ICusumChangepointFunc``.
- ``estimator`` -- Defines ``IEstimatingSchema`` and ``ISchemaEstimates``.
- ``generalized_cusum`` -- Defines ``GeneralizedCUSUM``,
  ``GeneralizedCUSUMConfiguration``, and ``GeneralizedCUSUMState``.
- ``monitoring`` -- Defines ``IMonitoringSchema``.

Examples
--------
>>> from pysatl_cpd.algorithms.online.cusum.algorithm import PageTwoSidedCusum
>>>
>>> cusum = PageTwoSidedCusum(
...     learning_period_size=30,
...     delta=0.5,
...     cov_reg=1e-6,
... )
>>> cusum.name
'PageTwoSidedCusum'
>>> cusum.configuration.adaptive_estimation
True
>>> cusum.state.is_in_learning_period
True
>>> values = [cusum.process(float(i)) for i in range(60)]
>>> cusum.state.is_in_learning_period
False
>>> cusum.reset()
>>> cusum.state.is_in_learning_period
True

Notes
-----
This subpackage contains only abstract interfaces and the generic
``GeneralizedCUSUM`` skeleton. Concrete algorithms (e.g.,
``PageTwoSidedCusum``, ``CrosierCusum``, ``AutoregressiveCUSUM``)
live in ``pysatl_cpd.algorithms.online.cusum.algorithm`` and supply
concrete component implementations from the ``component`` subpackage.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.cusum.abstracts.changepoint_func import ICusumChangepointFunc
from pysatl_cpd.algorithms.online.cusum.abstracts.estimator import IEstimatingSchema
from pysatl_cpd.algorithms.online.cusum.abstracts.generalized_cusum import GeneralizedCUSUM
from pysatl_cpd.algorithms.online.cusum.abstracts.monitoring import IMonitoringSchema

__all__ = ["IEstimatingSchema", "IMonitoringSchema", "ICusumChangepointFunc", "GeneralizedCUSUM"]
