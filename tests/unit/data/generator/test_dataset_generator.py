# -*- coding: ascii -*-
"""
Tests for dataset generator.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.data.generator.dataset import ScenarioDatasetGenerator
from pysatl_cpd.data.generator.specs import ScenarioSpec, SegmentPlan, SegmentSpec, UnivariateDistributionSpec
from pysatl_cpd.data.typedefs import StateDescriptor, frozendict


def _scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="mean_shift",
        segments=(SegmentSpec(plan_name="baseline", length=3), SegmentSpec(plan_name="shifted", length=2)),
        plans=frozendict.from_mapping(
            {
                "baseline": SegmentPlan(
                    distribution=_normal(mean=0.0, std=1.0), state=StateDescriptor(type="baseline")
                ),
                "shifted": SegmentPlan(distribution=_normal(mean=2.0, std=1.0), state=StateDescriptor(type="shifted")),
            }
        ),
    )


def test_scenario_dataset_generator_validates_constructor_and_generate_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ScenarioDatasetGenerator({})

    generator = ScenarioDatasetGenerator({"demo": _scenario()}, seed=42)

    with pytest.raises(ValueError, match="Dataset size"):
        generator.generate("demo", 0)
    with pytest.raises(ValueError, match="Unknown scenario"):
        generator.generate("missing", 1)


def test_scenario_dataset_generator_generates_named_dataset_with_metadata() -> None:
    generator = ScenarioDatasetGenerator({"demo": _scenario()}, seed=42)

    dataset = generator.generate("demo", 2)

    assert len(dataset) == 2
    assert dataset[0].name == "demo_series_0000"
    assert dataset[1].name == "demo_series_0001"
    assert dataset[0].annotation.metadata["scenario"] == "demo"
    assert dataset[0].change_points == (3,)
    assert dataset[0].raw_data.shape == (5, 1)


def _normal(*, mean: float = 0.0, std: float = 1.0) -> UnivariateDistributionSpec:
    return UnivariateDistributionSpec("Normal", "meanStd", mu=mean, sigma=std)
