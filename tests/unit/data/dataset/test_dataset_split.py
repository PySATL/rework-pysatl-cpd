# -*- coding: ascii -*-
"""
Tests for dataset split.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.data import Dataset
from pysatl_cpd.data.generator.dataset import ScenarioDatasetGenerator
from pysatl_cpd.data.generator.specs import NormalSpec, ScenarioSpec, SegmentPlan, SegmentSpec
from pysatl_cpd.data.typedefs import frozendict
from tests.support.providers import make_univariate_labeled


def test_dataset_train_test_split_without_random_state_uses_leading_items_for_test() -> None:
    dataset = Dataset(
        [
            make_univariate_labeled(name="series-0"),
            make_univariate_labeled(name="series-1"),
            make_univariate_labeled(name="series-2"),
            make_univariate_labeled(name="series-3"),
        ]
    )

    train, test = dataset.train_test_split(test_size=0.5, random_state=None)

    assert [provider.name for provider in test] == ["series-0", "series-1"]
    assert [provider.name for provider in train] == ["series-2", "series-3"]


def test_scenario_dataset_generator_scenarios_returns_copy() -> None:
    generator = ScenarioDatasetGenerator(
        {
            "demo": ScenarioSpec(
                name="demo",
                segments=(SegmentSpec(plan_name="baseline", length=2),),
                plans=frozendict.from_mapping({"baseline": SegmentPlan(distribution=NormalSpec())}),
            )
        }
    )

    scenarios = generator.scenarios
    scenarios["other"] = scenarios["demo"]

    assert set(generator.scenarios) == {"demo"}
