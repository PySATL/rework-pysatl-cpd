# -*- coding: ascii -*-
"""
Tests for extra coverage.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import replace

import numpy as np
import pytest

from pysatl_cpd.data.generator import GenericSeriesGenerator
from pysatl_cpd.data.generator.models import GeneratedSeries
from pysatl_cpd.data.generator.providers import build_plain_univariate_labeled_data
from pysatl_cpd.data.generator.segments.models import GeneratedSegment
from pysatl_cpd.data.generator.segments.sampling import sample_distribution
from pysatl_cpd.data.generator.specs import MultivariateNormalSpec
from pysatl_cpd.data.typedefs import SegmentInfo, StateDescriptor, frozendict


def test_build_plain_univariate_labeled_data_rejects_unknown_feature_name() -> None:
    series = GeneratedSeries(
        name="series",
        feature_names=("value",),
        data=np.asarray([[1.0], [2.0]], dtype=np.float64),
        segments=(
            SegmentInfo(
                segment_num=0,
                segment_start=0,
                segment_end=1,
                state=StateDescriptor(label="baseline"),
            ),
        ),
        metadata=frozendict(),
    )

    with pytest.raises(ValueError, match="Unknown feature name 'missing'"):
        build_plain_univariate_labeled_data(series, feature_name="missing", name="provider")


def test_sample_distribution_reshapes_one_dimensional_multivariate_output() -> None:
    class _StubRng:
        def multivariate_normal(self, *, mean, cov, size):
            del mean, cov, size
            return np.asarray([1.0, 2.0], dtype=np.float64)

    sampled = sample_distribution(
        MultivariateNormalSpec(means=frozendict.from_mapping({"x": 0.0}), covariance=1.0),
        2,
        _StubRng(),
    )

    assert sampled.shape == (2, 1)


def test_generate_from_segment_generators_uses_segment_type_when_state_is_empty() -> None:
    generator = GenericSeriesGenerator(seed=42)

    generated = generator.generate_from_segment_generators(
        [("baseline", _StubSegmentGenerator(feature_names=("x",), values=[[1.0], [2.0]]))]
    )

    assert generated.segments[0].state == StateDescriptor(type="baseline")


def test_generate_from_segment_generators_rejects_missing_feature_names_when_validation_is_bypassed() -> None:
    generator = GenericSeriesGenerator(seed=42)

    with pytest.raises(ValueError, match="must not be empty"):
        generator.generate_from_segment_generators([("baseline", _BrokenSegmentGenerator())])


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


class _BrokenSegmentGenerator:
    length = 1

    def generate(self, rng=None) -> GeneratedSegment:
        del rng
        segment = GeneratedSegment(
            name="segment",
            feature_names=("x",),
            data=np.asarray([[1.0]], dtype=np.float64),
            segment_info=SegmentInfo(
                segment_num=0,
                segment_start=0,
                segment_end=0,
                state=StateDescriptor(label="baseline"),
            ),
            metadata=frozendict(),
        )
        return replace(segment, feature_names=None)  # type: ignore[arg-type]
