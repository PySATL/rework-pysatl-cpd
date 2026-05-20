# -*- coding: ascii -*-
"""
Tests for presets.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.data.generator import preset_dataset, presets
from pysatl_cpd.data.generator.presets import (
    _build_segment_lengths,
    _correlated_covariance,
    _distribution_for_segment_type,
    _multivariate_distribution_for_segment_type,
    _segment_type_for_index,
    build_preset_scenario,
    get_preset_scenario,
)
from pysatl_cpd.data.generator.specs import IndependentColumnsSpec, MultivariateNormalSpec, ScenarioSpec
from pysatl_cpd.data.typedefs import frozendict


def test_build_preset_scenario_validates_inputs() -> None:
    with pytest.raises(ValueError, match="Series length"):
        build_preset_scenario("mean_shifts", feature_names=("x",), series_length=0, num_segments=3)
    with pytest.raises(ValueError, match="Number of segments"):
        build_preset_scenario("mean_shifts", feature_names=("x",), series_length=10, num_segments=0)
    with pytest.raises(ValueError, match="Feature names"):
        build_preset_scenario("mean_shifts", feature_names=(), series_length=10, num_segments=3)


@pytest.mark.parametrize(
    ("preset", "expected_types"),
    [
        ("mean_shifts", ("baseline", "alternative_1", "alternative_2")),
        ("no_shifts", ("baseline", "baseline", "baseline")),
        ("mixed_shifts", ("baseline", "mean_shift", "variance_shift")),
        ("extreme_mean_shifts", ("baseline", "extreme_high", "extreme_low")),
    ],
)
def test_build_preset_scenario_uses_expected_segment_types(preset: str, expected_types: tuple[str, ...]) -> None:
    scenario = build_preset_scenario(preset, feature_names=("x", "y"), series_length=12, num_segments=3)

    assert tuple(segment.plan_name for segment in scenario.segments) == expected_types
    assert tuple(plan.state["type"] for plan in scenario.plans.values()) == tuple(dict.fromkeys(expected_types))


def test_distribution_for_segment_type_covers_main_branches() -> None:
    assert isinstance(
        _distribution_for_segment_type(preset="mean_shifts", segment_type="baseline", feature_names=("x", "y")),
        MultivariateNormalSpec,
    )
    assert isinstance(
        _distribution_for_segment_type(preset="mixed_shifts", segment_type="baseline", feature_names=("x", "y")),
        IndependentColumnsSpec,
    )
    variance_shift = _distribution_for_segment_type(
        preset="mixed_shifts",
        segment_type="variance_shift",
        feature_names=("x", "y"),
    )
    assert isinstance(variance_shift, MultivariateNormalSpec)
    with pytest.raises(ValueError, match="Unsupported preset"):
        _distribution_for_segment_type(preset="unknown", segment_type="baseline", feature_names=("x",))


@pytest.mark.parametrize(
    ("segment_type", "expected_mean", "expected_covariance"),
    [
        ("baseline", 0.0, (1.0, 1.0)),
        ("alternative_1", 2.0, (1.0, 1.0)),
        ("alternative_2", -2.0, (1.0, 1.0)),
    ],
)
def test_distribution_for_mean_shifts_segments(
    segment_type: str,
    expected_mean: float,
    expected_covariance: tuple[float, float],
) -> None:
    distribution = _distribution_for_segment_type(
        preset="mean_shifts",
        segment_type=segment_type,
        feature_names=("x", "y"),
    )

    assert distribution.means == frozendict.from_mapping({"x": expected_mean, "y": expected_mean})
    assert distribution.covariance == expected_covariance


@pytest.mark.parametrize(
    ("segment_type", "expected_covariance"),
    [
        ("baseline", (1.0, 1.0)),
        ("alternative_1", (3.0, 3.0)),
        ("alternative_2", (0.25, 0.25)),
    ],
)
def test_distribution_for_variance_shifts_segments(
    segment_type: str,
    expected_covariance: tuple[float, float],
) -> None:
    distribution = _distribution_for_segment_type(
        preset="variance_shifts",
        segment_type=segment_type,
        feature_names=("x", "y"),
    )

    assert distribution.means == frozendict.from_mapping({"x": 0.0, "y": 0.0})
    assert distribution.covariance == expected_covariance


@pytest.mark.parametrize(
    ("segment_type", "expected_covariance"),
    [
        ("baseline", (1.0, 1.0)),
        ("alternative_1", ((1.0, 0.5), (0.5, 1.0))),
        ("alternative_2", ((1.0, -0.5), (-0.5, 1.0))),
    ],
)
def test_distribution_for_covariance_shifts_segments(
    segment_type: str,
    expected_covariance: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]],
) -> None:
    distribution = _distribution_for_segment_type(
        preset="covariance_shifts",
        segment_type=segment_type,
        feature_names=("x", "y"),
    )

    assert distribution.means == frozendict.from_mapping({"x": 0.0, "y": 0.0})
    assert distribution.covariance == expected_covariance


def test_preset_dataset_builds_named_dataset_with_default_feature_count() -> None:
    dataset = preset_dataset("3d_mean_shifts", n_series=2, seed=42, series_length=12, num_segments=3)

    assert len(dataset) == 2
    assert dataset[0].name == "3d_mean_shifts_series_0000"
    assert dataset[0].feature_columns == ["feature_0", "feature_1", "feature_2"]


def test_preset_dataset_defaults_to_two_features_for_non_3d_presets() -> None:
    dataset = preset_dataset("mean_shifts", n_series=1, seed=42, series_length=12, num_segments=3)

    assert dataset[0].feature_columns == ["feature_0", "feature_1"]


def test_preset_dataset_accepts_explicit_feature_count() -> None:
    dataset = preset_dataset("mean_shifts", n_features=5, n_series=1, seed=42, series_length=12, num_segments=3)

    assert dataset[0].feature_columns == ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]


def test_helper_functions_cover_segment_lengths_types_and_covariance() -> None:
    assert _build_segment_lengths(series_length=10, num_segments=3) == [4, 3, 3]
    assert _segment_type_for_index("mixed_shifts", 2) == "variance_shift"
    covariance = _correlated_covariance(3, 0.25)
    assert covariance[0][0] == 1.0
    assert covariance[0][1] == 0.25
    with pytest.raises(ValueError, match="Unknown preset scenario"):
        get_preset_scenario("missing")


@pytest.mark.parametrize(
    ("segment_index", "expected_segment_type"),
    [
        (0, "baseline"),
        (1, "extreme_high"),
        (2, "extreme_low"),
        (4, "extreme_high"),
    ],
)
def test_segment_type_for_index_handles_extreme_mean_shifts(
    segment_index: int,
    expected_segment_type: str,
) -> None:
    assert _segment_type_for_index("extreme_mean_shifts", segment_index) == expected_segment_type


def test_segment_type_for_index_uses_default_alternatives_for_unknown_presets() -> None:
    assert _segment_type_for_index("something_else", 0) == "baseline"
    assert _segment_type_for_index("something_else", 1) == "alternative_1"
    assert _segment_type_for_index("something_else", 2) == "alternative_2"


def test_multivariate_distribution_for_segment_type_rejects_unknown_presets() -> None:
    with pytest.raises(ValueError, match="Unsupported preset"):
        _multivariate_distribution_for_segment_type(
            preset="unknown",
            segment_type="baseline",
            feature_names=("x", "y"),
        )


def test_get_preset_scenario_returns_known_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = ScenarioSpec(
        name="known",
        segments=build_preset_scenario(
            "mean_shifts",
            feature_names=("x", "y"),
            series_length=12,
            num_segments=3,
        ).segments,
        plans=build_preset_scenario(
            "mean_shifts",
            feature_names=("x", "y"),
            series_length=12,
            num_segments=3,
        ).plans,
        metadata=frozendict(source="test"),
    )
    monkeypatch.setattr(
        presets,
        "PRESET_SCENARIOS",
        frozendict.from_mapping({"known": scenario}),
    )

    assert get_preset_scenario("known") is scenario
