# -*- coding: ascii -*-
"""
Tests for specs.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.data.generator.specs import (
    IndependentColumnsSpec,
    ScenarioSpec,
    SegmentPlan,
    SegmentSpec,
    UnivariateDistributionSpec,
    freeze_distribution_mapping,
    freeze_float_mapping,
    freeze_state_mapping,
    freeze_univariate_mapping,
)
from pysatl_cpd.data.typedefs import StateDescriptor, frozendict


def test_multivariate_normal_spec_rejects_empty_means() -> None:
    from pysatl_cpd.data.generator.specs import MultivariateNormalSpec

    with pytest.raises(ValueError, match="means must not be empty"):
        MultivariateNormalSpec()


def test_independent_columns_spec_rejects_empty_columns() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        IndependentColumnsSpec()


def test_segment_spec_rejects_empty_plan_name() -> None:
    with pytest.raises(ValueError, match="plan name must not be empty"):
        SegmentSpec(plan_name="", length=1)


def test_segment_spec_rejects_non_positive_length() -> None:
    with pytest.raises(ValueError, match="length must be positive"):
        SegmentSpec(plan_name="baseline", length=0)


def test_scenario_spec_rejects_empty_name() -> None:
    plan = SegmentPlan(distribution=_normal())

    with pytest.raises(ValueError, match="Scenario name must not be empty"):
        ScenarioSpec(
            name="",
            segments=(SegmentSpec(plan_name="baseline", length=1),),
            plans=frozendict.from_mapping({"baseline": plan}),
        )


def test_scenario_spec_rejects_empty_segments() -> None:
    plan = SegmentPlan(distribution=_normal())

    with pytest.raises(ValueError, match="at least one segment"):
        ScenarioSpec(name="scenario", segments=(), plans=frozendict.from_mapping({"baseline": plan}))


def test_scenario_spec_rejects_empty_plans() -> None:
    with pytest.raises(ValueError, match="plans must not be empty"):
        ScenarioSpec(name="scenario", segments=(SegmentSpec(plan_name="baseline", length=1),), plans=frozendict())


def test_freeze_float_mapping_converts_values_to_float() -> None:
    frozen = freeze_float_mapping({"x": 1, "y": 2.5})

    assert frozen == {"x": 1.0, "y": 2.5}


def test_freeze_state_mapping_returns_state_descriptor() -> None:
    state = freeze_state_mapping({"kind": "baseline", "flag": True})

    assert isinstance(state, StateDescriptor)
    assert dict(state) == {"kind": "baseline", "flag": True}


def test_freeze_univariate_mapping_returns_frozendict() -> None:
    frozen = freeze_univariate_mapping({"baseline": _normal(mean=1.0)})

    assert isinstance(frozen, frozendict)
    assert frozen["baseline"] == _normal(mean=1.0)


def test_univariate_distribution_spec_rejects_empty_family() -> None:
    with pytest.raises(ValueError, match="family must not be empty"):
        UnivariateDistributionSpec("", "meanStd", mu=0.0, sigma=1.0)


def test_univariate_distribution_spec_rejects_empty_parameter_name() -> None:
    with pytest.raises(ValueError, match="parameter names must not be empty"):
        UnivariateDistributionSpec("Normal", "meanStd", **{"": 1.0})


def test_freeze_distribution_mapping_returns_frozendict() -> None:
    plan_distribution = IndependentColumnsSpec(columns=frozendict.from_mapping({"x": _normal()}))

    frozen = freeze_distribution_mapping({"plan": plan_distribution})

    assert isinstance(frozen, frozendict)
    assert frozen["plan"] == plan_distribution


def _normal(*, mean: float = 0.0, std: float = 1.0) -> UnivariateDistributionSpec:
    return UnivariateDistributionSpec("Normal", "meanStd", mu=mean, sigma=std)
