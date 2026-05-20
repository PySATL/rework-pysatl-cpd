# -*- coding: ascii -*-
"""
Tests for series.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.data.generator import GenericSeriesGenerator
from pysatl_cpd.data.generator.providers.pd_provider import build_pandas_univariate_labeled_data
from pysatl_cpd.data.generator.segments.models import GeneratedSegment
from pysatl_cpd.data.generator.specs import (
    IndependentColumnsSpec,
    ScenarioSpec,
    SegmentPlan,
    SegmentSpec,
    UnivariateDistributionSpec,
)
from pysatl_cpd.data.typedefs import SegmentInfo, StateDescriptor, frozendict


def test_generate_direct_univariate_distribution_without_independent_columns() -> None:
    scenario = ScenarioSpec(
        name="mean_shift",
        segments=(SegmentSpec(plan_name="baseline", length=3), SegmentSpec(plan_name="shifted", length=2)),
        plans=frozendict.from_mapping(
            {
                "baseline": SegmentPlan(
                    distribution=_normal(mean=0.0, std=1.0),
                    state=StateDescriptor(type="baseline"),
                ),
                "shifted": SegmentPlan(
                    distribution=_normal(mean=2.0, std=1.0),
                    state=StateDescriptor(type="shifted"),
                ),
            }
        ),
    )

    generated = GenericSeriesGenerator(seed=42).generate_from_scenario(scenario, name="series")

    assert generated.name == "series"
    assert generated.feature_names == ("value",)
    assert generated.data.shape == (5, 1)
    assert generated.change_points == (2,)
    assert [segment.segment_start for segment in generated.segments] == [0, 3]
    assert [segment.segment_end for segment in generated.segments] == [2, 4]


def test_reused_plan_name_creates_multiple_segments() -> None:
    scenario = ScenarioSpec(
        name="reused",
        segments=(
            SegmentSpec(plan_name="baseline", length=2),
            SegmentSpec(plan_name="shifted", length=2),
            SegmentSpec(plan_name="baseline", length=2),
        ),
        plans=frozendict.from_mapping(
            {
                "baseline": SegmentPlan(distribution=_normal(mean=0.0, std=1.0)),
                "shifted": SegmentPlan(distribution=_normal(mean=1.0, std=1.0)),
            }
        ),
    )

    generated = GenericSeriesGenerator(seed=42).generate_from_scenario(scenario)

    assert generated.change_points == (1, 3)
    assert [segment.state["type"] for segment in generated.segments] == ["baseline", "shifted", "baseline"]


def test_scenario_requires_referenced_plan() -> None:
    with pytest.raises(ValueError, match="missing segment plans"):
        ScenarioSpec(
            name="invalid",
            segments=(SegmentSpec(plan_name="missing", length=1),),
            plans=frozendict.from_mapping({"baseline": SegmentPlan(distribution=_normal())}),
        )


def test_generate_from_scenario_rejects_mismatched_feature_names() -> None:
    scenario = ScenarioSpec(
        name="mixed_features",
        segments=(SegmentSpec(plan_name="baseline", length=2), SegmentSpec(plan_name="shifted", length=2)),
        plans=frozendict.from_mapping(
            {
                "baseline": SegmentPlan(distribution=_normal()),
                "shifted": SegmentPlan(
                    distribution=IndependentColumnsSpec(
                        columns=frozendict.from_mapping({"x": _normal(), "y": _normal()})
                    )
                ),
            }
        ),
    )

    with pytest.raises(ValueError, match="same feature names and order"):
        GenericSeriesGenerator(seed=42).generate_from_scenario(scenario)


def test_generate_from_scenario_rejects_empty_segments_when_validation_is_bypassed() -> None:
    scenario = object.__new__(ScenarioSpec)
    object.__setattr__(scenario, "name", "empty")
    object.__setattr__(scenario, "segments", ())
    object.__setattr__(scenario, "plans", frozendict.from_mapping({"baseline": SegmentPlan(distribution=_normal())}))
    object.__setattr__(scenario, "metadata", frozendict())

    with pytest.raises(ValueError, match="at least one segment"):
        GenericSeriesGenerator(seed=42).generate_from_scenario(scenario)


def test_generate_from_segment_generators_rejects_empty_generators() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        GenericSeriesGenerator(seed=42).generate_from_segment_generators([])


def test_generate_from_segment_generators_rejects_mismatched_feature_names() -> None:
    with pytest.raises(ValueError, match="same feature names and order"):
        GenericSeriesGenerator(seed=42).generate_from_segment_generators(
            [
                ("baseline", _StubSegmentGenerator(feature_names=("x",), values=[[1.0]])),
                ("shifted", _StubSegmentGenerator(feature_names=("y",), values=[[2.0]])),
            ]
        )


def test_build_pandas_univariate_labeled_data_selects_one_column() -> None:
    scenario = ScenarioSpec(
        name="univariate",
        segments=(SegmentSpec(plan_name="baseline", length=3),),
        plans=frozendict.from_mapping({"baseline": SegmentPlan(distribution=_normal())}),
    )
    generated = GenericSeriesGenerator(seed=42).generate_from_scenario(scenario)

    labeled = build_pandas_univariate_labeled_data(generated, feature_name="value", name="provider")

    assert labeled.feature_columns == ["value"]
    assert labeled.dataset()["value"].shape == (3,)
    assert np.asarray(list(labeled)).shape == (3,)


def test_build_pandas_univariate_labeled_data_rejects_unknown_feature() -> None:
    scenario = ScenarioSpec(
        name="univariate",
        segments=(SegmentSpec(plan_name="baseline", length=3),),
        plans=frozendict.from_mapping({"baseline": SegmentPlan(distribution=_normal())}),
    )
    generated = GenericSeriesGenerator(seed=42).generate_from_scenario(scenario)

    with pytest.raises(ValueError, match="Unknown feature name"):
        build_pandas_univariate_labeled_data(generated, feature_name="missing", name="provider")


class _StubSegmentGenerator:
    def __init__(self, *, feature_names: tuple[str, ...], values: list[list[float]]) -> None:
        self._feature_names = feature_names
        self._values = values

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def length(self) -> int:
        return len(self._values)

    def generate(self, rng=None) -> GeneratedSegment:
        del rng
        return GeneratedSegment(
            name="segment",
            feature_names=self._feature_names,
            data=np.asarray(self._values, dtype=np.float64),
            segment_info=SegmentInfo(
                segment_num=0,
                segment_start=0,
                segment_end=len(self._values) - 1,
                state=StateDescriptor(),
            ),
            metadata=frozendict(),
        )


def _normal(*, mean: float = 0.0, std: float = 1.0) -> UnivariateDistributionSpec:
    return UnivariateDistributionSpec("Normal", "meanStd", mu=mean, sigma=std)
