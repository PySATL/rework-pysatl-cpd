# -*- coding: ascii -*-
"""
Change-point statistics for the Symbolic Divergence algorithm.

This package provides the interface and concrete implementations that turn the
stream of divergence values into the final change-point statistic compared
against a threshold by the detector. It plays the same role as the CUSUM
change-point function.

.. raw:: html

    <h2>Public API</h2>

- ``IChangePointStatistic`` -- Protocol for divergence-to-statistic mappings.
  Any class implementing ``update``, ``value``, and ``reset`` is compatible.
- ``RawDivergenceStatistic`` -- Identity statistic returning the latest
  divergence unchanged (default).
- ``ScaledDivergenceStatistic`` -- ``scale * sample_size * divergence``; with
  ``scale = 2.0`` this reproduces the ``2 n D`` statistic.
- ``LogScaledDivergenceStatistic`` -- ``scale * sample_size /
  log(sample_size + 1) * divergence``; a logarithmically damped variant of the
  linear scaling.
"""

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.base import IChangePointStatistic
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.log_scaled import LogScaledDivergenceStatistic
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.raw import RawDivergenceStatistic
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.scaled import ScaledDivergenceStatistic

__all__ = [
    "IChangePointStatistic",
    "LogScaledDivergenceStatistic",
    "RawDivergenceStatistic",
    "ScaledDivergenceStatistic",
]
