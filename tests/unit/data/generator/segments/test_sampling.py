# -*- coding: ascii -*-
"""
Tests for sampling.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import numpy as np
import pytest

from pysatl_cpd.data.generator.segments.sampling import (
    DEFAULT_UNIVARIATE_FEATURE_NAME,
    build_covariance_matrix,
    feature_names_for_distribution,
    sample_distribution,
    sample_univariate_distribution,
)
from pysatl_cpd.data.generator.specs import (
    ExponentialSpec,
    IndependentColumnsSpec,
    MultivariateNormalSpec,
    NormalSpec,
    StudentTSpec,
    UniformSpec,
)
from pysatl_cpd.data.typedefs import frozendict


def test_feature_names_for_distribution_covers_supported_specs() -> None:
    assert feature_names_for_distribution(NormalSpec()) == (DEFAULT_UNIVARIATE_FEATURE_NAME,)
    assert feature_names_for_distribution(
        MultivariateNormalSpec(means=frozendict.from_mapping({"x": 0.0, "y": 1.0}))
    ) == (
        "x",
        "y",
    )
    assert feature_names_for_distribution(
        IndependentColumnsSpec(columns=frozendict.from_mapping({"x": NormalSpec(), "y": UniformSpec()}))
    ) == ("x", "y")


def test_feature_names_for_distribution_rejects_unsupported_spec_type() -> None:
    unsupported: Any = object()

    with pytest.raises(TypeError, match="Unsupported distribution spec type"):
        feature_names_for_distribution(unsupported)


def test_sample_univariate_distribution_covers_all_branches() -> None:
    rng = np.random.default_rng(42)
    assert sample_univariate_distribution(NormalSpec(mean=1.0, std=2.0), 3, rng).shape == (3,)
    assert sample_univariate_distribution(UniformSpec(low=-1.0, high=1.0), 3, rng).shape == (3,)
    assert sample_univariate_distribution(ExponentialSpec(scale=2.0), 3, rng).shape == (3,)
    assert sample_univariate_distribution(StudentTSpec(df=5.0, loc=1.0, scale=2.0), 3, rng).shape == (3,)


def test_sample_distribution_covers_univariate_multivariate_and_independent_columns() -> None:
    rng = np.random.default_rng(42)
    univariate = sample_distribution(NormalSpec(), 4, rng)
    multivariate = sample_distribution(
        MultivariateNormalSpec(means=frozendict.from_mapping({"x": 0.0, "y": 1.0}), covariance=(1.0, 2.0)),
        4,
        rng,
    )
    independent = sample_distribution(
        IndependentColumnsSpec(columns=frozendict.from_mapping({"x": NormalSpec(), "y": UniformSpec()})),
        4,
        rng,
    )

    assert univariate.shape == (4, 1)
    assert multivariate.shape == (4, 2)
    assert independent.shape == (4, 2)


def test_sample_distribution_reshapes_single_feature_multivariate_result() -> None:
    rng = np.random.default_rng(42)

    sampled = sample_distribution(
        MultivariateNormalSpec(means=frozendict.from_mapping({"x": 0.0}), covariance=1.0),
        1,
        rng,
    )

    assert sampled.shape == (1, 1)


def test_sample_distribution_rejects_unsupported_spec_type() -> None:
    rng = np.random.default_rng(42)
    unsupported: Any = object()

    with pytest.raises(TypeError, match="Unsupported distribution spec type"):
        sample_distribution(unsupported, 3, rng)


def test_sample_univariate_distribution_rejects_unsupported_spec_type() -> None:
    rng = np.random.default_rng(42)
    unsupported: Any = object()

    with pytest.raises(TypeError, match="Unsupported univariate distribution spec type"):
        sample_univariate_distribution(unsupported, 3, rng)


def test_build_covariance_matrix_covers_scalar_diagonal_full_and_errors() -> None:
    assert build_covariance_matrix(2.0, 1).tolist() == [[2.0]]
    assert build_covariance_matrix((1.0, 2.0), 2).tolist() == [[1.0, 0.0], [0.0, 2.0]]
    assert build_covariance_matrix(((1.0, 0.5), (0.5, 2.0)), 2).tolist() == [[1.0, 0.5], [0.5, 2.0]]

    with pytest.raises(ValueError, match="only valid for one feature"):
        build_covariance_matrix(1.0, 2)
    with pytest.raises(ValueError, match="Covariance diagonal"):
        build_covariance_matrix((1.0,), 2)
    with pytest.raises(ValueError, match="Covariance matrix"):
        build_covariance_matrix(((1.0, 0.0),), 2)
