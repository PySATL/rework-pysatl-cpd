# -*- coding: ascii -*-
"""Sampling helpers for generated segments."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np

from pysatl_cpd.data.generator.specs import (
    DistributionSpec,
    ExponentialSpec,
    IndependentColumnsSpec,
    MultivariateNormalSpec,
    NormalSpec,
    StudentTSpec,
    UniformSpec,
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
    if isinstance(distribution, NormalSpec | UniformSpec | ExponentialSpec | StudentTSpec):
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
    if isinstance(distribution, NormalSpec | UniformSpec | ExponentialSpec | StudentTSpec):
        return sample_univariate_distribution(distribution, length, rng).reshape(-1, 1)

    if isinstance(distribution, MultivariateNormalSpec):
        covariance = build_covariance_matrix(distribution.covariance, len(distribution.means))
        mean_vector = np.asarray(list(distribution.means.values()), dtype=np.float64)
        sampled = rng.multivariate_normal(mean=mean_vector, cov=covariance, size=length)
        if sampled.ndim == 1:
            sampled = sampled.reshape(-1, 1)
        return sampled

    if isinstance(distribution, IndependentColumnsSpec):
        columns = [sample_univariate_distribution(spec, length, rng) for spec in distribution.columns.values()]
        return np.column_stack(columns)

    raise TypeError(f"Unsupported distribution spec type: {type(distribution).__name__}")


def sample_univariate_distribution(
    distribution: NormalSpec | UniformSpec | ExponentialSpec | StudentTSpec,
    length: int,
    rng: np.random.Generator,
) -> NumericArray:
    """
    Sample from a univariate distribution specification.

    Parameters
    ----------
    distribution
        Univariate distribution specification.
    length
        Number of samples to generate.
    rng
        Random number generator for reproducible sampling.

    Returns
    -------
    samples
        Array of sampled values.

    Raises
    ------
    TypeError
        If the distribution type is not supported.
    """
    if isinstance(distribution, NormalSpec):
        return rng.normal(loc=distribution.mean, scale=distribution.std, size=length)
    if isinstance(distribution, UniformSpec):
        return rng.uniform(low=distribution.low, high=distribution.high, size=length)
    if isinstance(distribution, ExponentialSpec):
        return rng.exponential(scale=distribution.scale, size=length)
    if isinstance(distribution, StudentTSpec):
        return rng.standard_t(df=distribution.df, size=length) * distribution.scale + distribution.loc
    raise TypeError(f"Unsupported univariate distribution spec type: {type(distribution).__name__}")


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
