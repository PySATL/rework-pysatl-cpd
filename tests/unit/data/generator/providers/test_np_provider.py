# -*- coding: ascii -*-
"""
Tests for np provider.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.data.generator.models import GeneratedSeries
from pysatl_cpd.data.generator.providers.np_provider import (
    build_plain_multivariate_labeled_data,
    build_plain_univariate_labeled_data,
)
from pysatl_cpd.data.typedefs import SegmentInfo, StateDescriptor, TimeseriesAnnotation
from pysatl_cpd.typedefs import frozendict


def _make_generated_series() -> GeneratedSeries:
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    segments = (
        SegmentInfo(segment_num=0, segment_start=0, segment_end=1, state=StateDescriptor(type="a")),
        SegmentInfo(segment_num=1, segment_start=2, segment_end=2, state=StateDescriptor(type="b")),
    )
    return GeneratedSeries(
        name="test-series",
        data=data,
        segments=segments,
        feature_names=("f1", "f2"),
        metadata=frozendict(),
    )


class TestBuildPlainMultivariateLabeledData:
    def test_builds_with_explicit_annotation(self) -> None:
        series = _make_generated_series()
        annotation = TimeseriesAnnotation(name="explicit", metadata={"key": "value"})
        result = build_plain_multivariate_labeled_data(series, name="ignored", annotation=annotation)
        assert result.name == "explicit"
        assert result.annotation is annotation

    def test_builds_creates_annotation_when_none(self) -> None:
        series = _make_generated_series()
        result = build_plain_multivariate_labeled_data(series, name="auto-name")
        assert result.name == "auto-name"


class TestBuildPlainUnivariateLabeledData:
    def test_raises_for_unknown_feature(self) -> None:
        series = _make_generated_series()
        with pytest.raises(ValueError, match="Unknown feature name"):
            build_plain_univariate_labeled_data(series, feature_name="unknown", name="test")

    def test_builds_with_explicit_annotation(self) -> None:
        series = _make_generated_series()
        annotation = TimeseriesAnnotation(name="explicit", metadata={"key": "value"})
        result = build_plain_univariate_labeled_data(series, feature_name="f1", name="ignored", annotation=annotation)
        assert result.name == "explicit"
        assert result.annotation is annotation

    def test_builds_creates_annotation_when_none(self) -> None:
        series = _make_generated_series()
        result = build_plain_univariate_labeled_data(series, feature_name="f2", name="auto-name")
        assert result.name == "auto-name"
        assert len(result) == 3
