# -*- coding: ascii -*-
"""
Tests for segment series generation.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.data.generator.segments.models import GeneratedSegment
from pysatl_cpd.data.generator.series import GenericSeriesGenerator
from pysatl_cpd.data.typedefs import SegmentInfo, StateDescriptor, frozendict


class _StubSegmentGenerator:
    def __init__(
        self, *, feature_names: tuple[str, ...], values: list[list[float]], state_type: str | None = None
    ) -> None:
        self._feature_names = feature_names
        self._values = values
        self._state_type = state_type

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def length(self) -> int:
        return len(self._values)

    def generate(self, rng=None) -> GeneratedSegment:
        del rng
        state = StateDescriptor(type=self._state_type) if self._state_type is not None else StateDescriptor()
        return GeneratedSegment(
            name="segment",
            feature_names=self._feature_names,
            data=np.asarray(self._values, dtype=np.float64),
            segment_info=SegmentInfo(segment_num=0, segment_start=0, segment_end=len(self._values) - 1, state=state),
            metadata=frozendict(),
        )


def test_generate_from_segment_generators_builds_series_and_fallback_state() -> None:
    generator = GenericSeriesGenerator(seed=42)
    series = generator.generate_from_segment_generators(
        [
            ("baseline", _StubSegmentGenerator(feature_names=("x",), values=[[1.0], [2.0]], state_type=None)),
            ("shift", _StubSegmentGenerator(feature_names=("x",), values=[[3.0]], state_type="shift")),
        ],
        name="series",
    )

    assert series.name == "series"
    assert series.feature_names == ("x",)
    assert series.change_points == (1,)
    assert [dict(segment.state) for segment in series.segments] == [{"type": "baseline"}, {"type": "shift"}]


def test_generate_from_segment_generators_rejects_empty_or_mismatched_features() -> None:
    generator = GenericSeriesGenerator(seed=42)

    with pytest.raises(ValueError, match="must not be empty"):
        generator.generate_from_segment_generators([])

    with pytest.raises(ValueError, match="same feature names"):
        generator.generate_from_segment_generators(
            [
                ("baseline", _StubSegmentGenerator(feature_names=("x",), values=[[1.0]])),
                ("shift", _StubSegmentGenerator(feature_names=("y",), values=[[2.0]])),
            ]
        )
