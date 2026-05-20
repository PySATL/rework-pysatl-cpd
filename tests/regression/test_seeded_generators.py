# -*- coding: ascii -*-
"""
Tests for seeded generators.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pathlib import Path

import numpy as np
import pytest

from pysatl_cpd.data.generator import GenericSeriesGenerator
from pysatl_cpd.data.generator.specs import ScenarioSpec, SegmentPlan, SegmentSpec, UnivariateDistributionSpec
from pysatl_cpd.data.typedefs import StateDescriptor, frozendict
from tests.support.golden import load_json_golden


@pytest.mark.xfail(reason="pysatl-core UNURAN sampling does not expose seed control yet", strict=False)
def test_generated_series_matches_golden() -> None:
    golden = load_json_golden(Path(__file__).resolve().parents[1] / "golden" / "generators" / "simple_series.json")
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

    assert generated.name == golden["name"]
    assert list(generated.feature_names) == golden["feature_names"]
    assert generated.change_points == tuple(golden["change_points"])
    assert [
        (segment.segment_num, segment.segment_start, segment.segment_end, dict(segment.state))
        for segment in generated.segments
    ] == [tuple(row[:3]) + (row[3],) for row in golden["segments"]]
    assert np.allclose(generated.data, np.asarray(golden["data"], dtype=np.float64))


def _normal(*, mean: float = 0.0, std: float = 1.0) -> UnivariateDistributionSpec:
    return UnivariateDistributionSpec("Normal", "meanStd", mu=mean, sigma=std)
