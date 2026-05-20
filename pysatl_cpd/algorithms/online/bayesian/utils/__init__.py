# -*- coding: ascii -*-
"""Factory helpers for Bayesian online change-point detection components.

This module provides factory functions that construct the three core building
blocks of a Bayesian online change-point detection system: hazard functions,
likelihood models, and change-point score functions (CPF). Each factory
returns an object conforming to the corresponding protocol defined in the
sibling ``protocol`` subpackage.

The concrete implementations live in the ``component`` subpackage. Use these
factories to obtain correctly configured instances without importing the
implementation classes directly.

.. raw:: html

    <h2>Public API</h2>

- ``get_hazard_function(lambda_)`` -> ``IHazard``
  Builds a constant hazard model with the given expected run length.
- ``get_likelihood_function(mu_0, k_0, alpha_0, beta_0)`` -> ``ILikelihood``
  Builds a Gaussian conjugate likelihood with the specified prior parameters.
- ``get_cpf_function(cpf_type)`` -> ``IBayesianCPF``
  Builds a change-point score function selected by ``BayesianCPFType``.

Notes
-----
The ``BayesianCPFType`` enum is defined in ``pysatl_cpd.algorithms.online.bayesian._enum``
and re-exported from the parent ``bayesian`` package.

Examples
--------
Construct all three components for a univariate Gaussian BOCPD setup:

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
from pysatl_cpd.algorithms.online.bayesian.component.cpf import DropCPF, MaxRunLengthCPF
from pysatl_cpd.algorithms.online.bayesian.component.hazard import ConstantHazard
from pysatl_cpd.algorithms.online.bayesian.component.likelihood import GaussianConjugate
from pysatl_cpd.algorithms.online.bayesian.protocol.cpf import IBayesianCPF
from pysatl_cpd.algorithms.online.bayesian.protocol.hazard import IHazard
from pysatl_cpd.algorithms.online.bayesian.protocol.likelihood import ILikelihood


def get_hazard_function(lambda_: float) -> IHazard:
    """Construct a constant hazard model.

    Parameters
    ----------
    lambda_
        Expected run length (must be >= 1.0).

    Returns
    -------
    IHazard
    """
    return ConstantHazard(lambda_=lambda_)


def get_likelihood_function(mu_0: float, k_0: float, alpha_0: float, beta_0: float) -> ILikelihood:
    """Construct a Gaussian conjugate likelihood.

    Parameters
    ----------
    mu_0
        Prior mean.
    k_0
        Prior pseudo-count (must be > 0).
    alpha_0
        Prior shape (must be > 0).
    beta_0
        Prior scale (must be > 0).

    Returns
    -------
    ILikelihood
    """
    return GaussianConjugate(mu_0=mu_0, k_0=k_0, alpha_0=alpha_0, beta_0=beta_0)


def get_cpf_function(cpf_type: BayesianCPFType) -> IBayesianCPF:
    """Construct a Bayesian change-point score function.

    Parameters
    ----------
    cpf_type
        Enum selecting the CPF variant.

    Returns
    -------
    IBayesianCPF

    Raises
    ------
    ValueError
        If *cpf_type* is unknown.
    """
    if cpf_type == BayesianCPFType.MAX_RUN_LENGTH:
        return MaxRunLengthCPF()
    if cpf_type == BayesianCPFType.DROP:
        return DropCPF()
    raise ValueError(f"Unknown cpf_type: {cpf_type}")


__all__ = ["get_cpf_function", "get_hazard_function", "get_likelihood_function"]
