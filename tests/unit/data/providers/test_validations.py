# -*- coding: ascii -*-
"""
Tests for validations.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pandas as pd
import pytest

from pysatl_cpd.data import (
    Dataset,
    NDArrayMultivariateProvider,
    NDArrayUnivariateProvider,
    PandasLabeledData,
    PlainMultivariateLabeledData,
    PlainUnivariateLabeledData,
)
from pysatl_cpd.data.generator import ScenarioDatasetGenerator
from pysatl_cpd.data.providers.labeled.segments_labeling import SegmentsLabeling
from pysatl_cpd.data.providers.plain.pd_provider import PandasDataProvider
from pysatl_cpd.data.typedefs import SegmentInfo, StateDescriptor, TimeseriesAnnotation, UnlabeledTimeseriesAnnotation


def _unlabeled_annotation(name: str = "series") -> UnlabeledTimeseriesAnnotation:
    return UnlabeledTimeseriesAnnotation(name=name)


def _labeled_annotation(name: str = "series") -> TimeseriesAnnotation:
    return TimeseriesAnnotation(name=name)


def _segments() -> list[SegmentInfo]:
    return [
        SegmentInfo(segment_num=0, segment_start=0, segment_end=2, state=StateDescriptor(label="a")),
        SegmentInfo(segment_num=1, segment_start=3, segment_end=5, state=StateDescriptor(label="b")),
    ]


@pytest.mark.parametrize(
    ("provider", "start", "stop", "message"),
    [
        (NDArrayUnivariateProvider(np.arange(5.0), _unlabeled_annotation()), -1, 1, "non-negative"),
        (NDArrayUnivariateProvider(np.arange(5.0), _unlabeled_annotation()), 3, 2, "greater than or equal"),
        (NDArrayUnivariateProvider(np.arange(5.0), _unlabeled_annotation()), 0, 5, "exceeds data length"),
        (
            NDArrayMultivariateProvider(np.arange(10.0).reshape(5, 2), _unlabeled_annotation()),
            -1,
            1,
            "non-negative",
        ),
        (
            NDArrayMultivariateProvider(np.arange(10.0).reshape(5, 2), _unlabeled_annotation()),
            3,
            2,
            "greater than or equal",
        ),
        (
            NDArrayMultivariateProvider(np.arange(10.0).reshape(5, 2), _unlabeled_annotation()),
            0,
            5,
            "exceeds data length",
        ),
    ],
)
def test_numpy_providers_validate_cut_boundaries(provider: object, start: int, stop: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        provider.cut(start, stop)  # type: ignore[attr-defined]


def test_pandas_provider_select_columns_requires_known_nonempty_columns() -> None:
    provider = PandasDataProvider(pd.DataFrame({"a": [1.0], "b": [2.0]}), _unlabeled_annotation())

    with pytest.raises(ValueError, match="At least one column"):
        provider.select_columns([])

    with pytest.raises(ValueError, match="Unknown columns"):
        provider.select_columns(["missing"])


def test_pandas_provider_merge_requires_same_columns() -> None:
    left = PandasDataProvider(pd.DataFrame({"a": [1.0], "b": [2.0]}), _unlabeled_annotation("left"))
    right = PandasDataProvider(pd.DataFrame({"b": [2.0], "a": [1.0]}), _unlabeled_annotation("right"))

    with pytest.raises(ValueError, match="same columns and column order"):
        PandasDataProvider.merge([left, right])


def test_pandas_provider_create_feature_column_requires_non_empty_name() -> None:
    provider = PandasDataProvider(pd.DataFrame({"a": [1.0], "b": [2.0]}), _unlabeled_annotation())

    with pytest.raises(ValueError, match="non-empty"):
        provider.create_feature_column(name="", mapping=lambda row: row["a"] + row["b"])


def test_segments_labeling_validates_contiguity() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        SegmentsLabeling(
            [
                SegmentInfo(segment_num=0, segment_start=0, segment_end=1, state=StateDescriptor(label="a")),
                SegmentInfo(segment_num=1, segment_start=3, segment_end=4, state=StateDescriptor(label="b")),
            ]
        )


def test_segments_labeling_cut_reindexes_segments() -> None:
    labeling = SegmentsLabeling(_segments())

    result = labeling.cut(2, 4)

    assert [segment.segment_num for segment in result] == [0, 1]
    assert [(segment.segment_start, segment.segment_end) for segment in result] == [(0, 0), (1, 2)]


def test_plain_univariate_from_unlabeled_data_returns_concrete_provider() -> None:
    provider = NDArrayUnivariateProvider(np.arange(6.0), _unlabeled_annotation())

    labeled = PlainUnivariateLabeledData.from_unlabeled_data(provider, _segments(), _labeled_annotation())

    assert isinstance(labeled, PlainUnivariateLabeledData)


def test_ndarray_univariate_provider_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError, match="1-dimensional"):
        NDArrayUnivariateProvider(np.arange(6.0).reshape(3, 2), _unlabeled_annotation())


def test_ndarray_univariate_provider_raw_data_returns_copy() -> None:
    provider = NDArrayUnivariateProvider(np.arange(3.0), _unlabeled_annotation())

    raw_data = provider.raw_data
    raw_data[0] = 99.0

    assert provider.raw_data.tolist() == [0.0, 1.0, 2.0]


def test_plain_multivariate_from_unlabeled_data_returns_concrete_provider() -> None:
    provider = NDArrayMultivariateProvider(np.arange(12.0).reshape(6, 2), _unlabeled_annotation())

    labeled = PlainMultivariateLabeledData.from_unlabeled_data(provider, _segments(), _labeled_annotation())

    assert isinstance(labeled, PlainMultivariateLabeledData)


def test_ndarray_multivariate_provider_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="Expected 2 dimensions"):
        NDArrayMultivariateProvider(np.arange(3.0), _unlabeled_annotation())


def test_labeled_data_cut_uses_shared_boundary_validation() -> None:
    provider = NDArrayUnivariateProvider(np.arange(6.0), _unlabeled_annotation())
    labeled = PlainUnivariateLabeledData.from_unlabeled_data(provider, _segments(), _labeled_annotation())

    with pytest.raises(ValueError, match="exceeds data length"):
        labeled.cut(0, 6)


def test_bisegment_info_validates_change_point_boundaries() -> None:
    from pysatl_cpd.data.typedefs import BisegmentInfo, TransitionDescriptor

    with pytest.raises(ValueError, match="within the bisegment"):
        BisegmentInfo(
            bisegment_num=0,
            bisegment_start=0,
            bisegment_end=3,
            change_point=4,
            transition=TransitionDescriptor(
                curr_state=StateDescriptor(label="a"),
                next_state=StateDescriptor(label="b"),
            ),
        )


def test_data_package_exports_are_importable() -> None:
    assert Dataset is not None
    assert PandasLabeledData is not None
    assert ScenarioDatasetGenerator is not None
