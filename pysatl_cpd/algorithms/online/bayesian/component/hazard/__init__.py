# -*- coding: ascii -*-
"""Hazard models for Bayesian online change-point detection.

This subpackage provides hazard (changepoint prior) models used in Bayesian
online change-point detection. Hazard models define the probability that a
changepoint occurs at a given time step as a function of the current run
length (the number of observations since the last changepoint).

.. raw:: html

    <h2>Public API</h2>

- ``ConstantHazard`` -- constant hazard model with a fixed expected run length
  timescale. Implements the ``IHazard`` protocol.

.. raw:: html

    <h2>Submodules</h2>

- ``constant`` -- defines the ``ConstantHazard`` class. See its docstring for
  implementation details.

.. raw:: html

    <h2>Examples</h2>

Create a constant hazard model with an expected run length of 100 observations
and compute log-hazard and log-survival values for a set of run lengths:

>>> import numpy as np
>>> from pysatl_cpd.algorithms.online.bayesian.component.hazard import ConstantHazard
>>> hazard = ConstantHazard(lambda_=100.0)
>>> run_lengths = np.array([0, 1, 2, 3], dtype=np.intp)
>>> log_h, log_surv = hazard.hazard(run_lengths)
>>> log_h
array([-4.60517019, -4.60517019, -4.60517019, -4.60517019])
>>> log_surv
array([-0.01005034, -0.01005034, -0.01005034, -0.01005034])

The ``lambda_`` parameter represents the expected run length and must be at
least 1.0:

>>> ConstantHazard(lambda_=0.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
ValueError: lambda_ must be >= 1.0

Notes
-----
The constant hazard model corresponds to a geometric run-length distribution
with parameter 1 / lambda\_. This is the standard changepoint prior used in
the Adams & MacKay (2007) Bayesian online change-point detection algorithm.

The ``hazard`` method returns log-space values for numerical stability. Both
arrays are broadcast to match the shape of the input ``run_lengths`` array.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.bayesian.component.hazard.constant import ConstantHazard

__all__ = ["ConstantHazard"]
