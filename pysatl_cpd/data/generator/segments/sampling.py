# -*- coding: ascii -*-
"""Sampling helpers for generated segments."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
from pysatl_core.families.configuration import configure_families_register
from pysatl_core.families.registry import ParametricFamilyRegister

from pysatl_cpd.data.generator.specs import (
    DistributionSpec,
    IndependentColumnsSpec,
    MultivariateNormalSpec,
    UnivariateDistributionSpec,
)
from pysatl_cpd.data.typedefs import NumericArray

DEFAULT_UNIVARIATE_FEATURE_NAME = "value"


def feature_names_for_distribution(distribution: DistributionSpec) -> tuple[str, ...]:
    """
    Extract feature names from a distribution specification.

    Parameters
    ----------
    distribution
        Distribution specification to extract feature names from.

    Returns
    -------
    feature_names
        Tuple of feature names for the distribution.

    Raises
    ------
    TypeError
        If the distribution type is not supported.
    """
    if isinstance(distribution, UnivariateDistributionSpec):
        return (DEFAULT_UNIVARIATE_FEATURE_NAME,)
    if isinstance(distribution, MultivariateNormalSpec):
        return tuple(distribution.means)
    if isinstance(distribution, IndependentColumnsSpec):
        return tuple(distribution.columns)
    raise TypeError(f"Unsupported distribution spec type: {type(distribution).__name__}")


def sample_distribution(
    distribution: DistributionSpec,
    length: int,
    rng: np.random.Generator,
) -> NumericArray:
    """
    Sample data from a distribution specification.

    Parameters
    ----------
    distribution
        Distribution specification to sample from.
    length
        Number of samples to generate.
    rng
        Random number generator for reproducible sampling.

    Returns
    -------
    samples
        Array of sampled data with shape ``(length, num_features)``.

    Raises
    ------
    TypeError
        If the distribution type is not supported.
    """
    if isinstance(distribution, UnivariateDistributionSpec):
        return _sample_core_univariate_distribution(distribution, length).reshape(-1, 1)

    if isinstance(distribution, MultivariateNormalSpec):
        covariance = build_covariance_matrix(distribution.covariance, len(distribution.means))
        mean_vector = np.asarray(list(distribution.means.values()), dtype=np.float64)
        sampled = rng.multivariate_normal(mean=mean_vector, cov=covariance, size=length)
        if sampled.ndim == 1:
            sampled = sampled.reshape(-1, 1)
        return sampled

    if isinstance(distribution, IndependentColumnsSpec):
        columns = [_sample_core_univariate_distribution(spec, length) for spec in distribution.columns.values()]
        return np.column_stack(columns)

    raise TypeError(f"Unsupported distribution spec type: {type(distribution).__name__}")


def _sample_core_univariate_distribution(distribution: UnivariateDistributionSpec, length: int) -> NumericArray:
    """Sample a univariate distribution through pysatl-core."""
    configure_families_register()
    family = ParametricFamilyRegister.get(distribution.family)
    core_distribution = family.distribution(
        parametrization_name=distribution.parametrization_name,
        sampling_strategy=None,
        computation_strategy=None,
        **dict(distribution.parameters),
    )
    return np.asarray(core_distribution.sample(length), dtype=np.float64).reshape(length)


def build_covariance_matrix(
    covariance: NumericArray | tuple[tuple[float, ...], ...] | tuple[float, ...] | float,
    dimension: int,
) -> NumericArray:
    """
    Construct a covariance matrix from supported covariance inputs.

    Parameters
    ----------
    covariance
        Covariance as scalar, 1D diagonal, or full matrix.
    dimension
        Expected dimension of the covariance matrix.

    Returns
    -------
    covariance_matrix
        Properly shaped covariance matrix.

    Raises
    ------
    ValueError
        If the covariance input does not match the expected dimension.
    """
    array = np.asarray(covariance, dtype=np.float64)
    if array.ndim == 0:
        if dimension != 1:
            raise ValueError(f"Covariance scalar is only valid for one feature, got {dimension} features")
        return array.reshape(1, 1)
    if array.ndim == 1:
        if array.shape != (dimension,):
            raise ValueError(f"Covariance diagonal must have shape {(dimension,)}, got {array.shape}")
        return np.diag(array)
    if array.shape != (dimension, dimension):
        raise ValueError(f"Covariance matrix must have shape {(dimension, dimension)}, got {array.shape}")
    return array
