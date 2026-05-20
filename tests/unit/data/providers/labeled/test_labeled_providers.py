# -*- coding: ascii -*-
"""
Tests for labeled providers.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pandas as pd
import pytest

from pysatl_cpd.data import NDArrayUnivariateProvider
from pysatl_cpd.data.providers.labeled import (
    LabeledData,
    PandasLabeledData,
    PlainMultivariateLabeledData,
    PlainUnivariateLabeledData,
)
from pysatl_cpd.data.typedefs import SegmentInfo, StateDescriptor, TimeseriesAnnotation, UnlabeledTimeseriesAnnotation
from tests.support.providers import (
    make_multivariate_provider,
    make_pandas_labeled,
    make_segments,
    make_univariate_labeled,
    make_univariate_provider,
)


def test_labeled_data_from_unlabeled_data_preserves_unlabeled_provider() -> None:
    unlabeled = make_univariate_provider(name="base")
    annotation = TimeseriesAnnotation(name="base")

    labeled = LabeledData.from_unlabeled_data(unlabeled, make_segments(), annotation)

    assert labeled.unlabeled is unlabeled
    assert labeled.annotation == annotation


def test_plain_univariate_labeled_data_rejects_non_univariate_provider() -> None:
    with pytest.raises(TypeError, match="requires an NDArrayUnivariateProvider"):
        PlainUnivariateLabeledData.from_unlabeled_data(
            make_multivariate_provider(),
            make_segments(),
            TimeseriesAnnotation(name="series"),
        )


def test_plain_univariate_labeled_data_raw_data_exposes_unlabeled_copy() -> None:
    labeled = PlainUnivariateLabeledData.from_unlabeled_data(
        make_univariate_provider(),
        make_segments(),
        TimeseriesAnnotation(name="series"),
    )

    raw_data = labeled.raw_data
    raw_data[0] = 99.0

    assert labeled.raw_data.tolist()[0] == 1.0


def test_plain_multivariate_labeled_data_rejects_non_multivariate_provider() -> None:
    with pytest.raises(TypeError, match="requires an NDArrayMultivariateProvider"):
        PlainMultivariateLabeledData.from_unlabeled_data(
            make_univariate_provider(),
            make_segments(),
            TimeseriesAnnotation(name="series"),
        )


def test_pandas_labeled_data_rejects_non_pandas_provider() -> None:
    with pytest.raises(TypeError, match="requires a PandasDataProvider"):
        PandasLabeledData.from_unlabeled_data(
            NDArrayUnivariateProvider(np.array([1.0, 2.0]), UnlabeledTimeseriesAnnotation(name="series")),
            make_segments(lengths=(2,), labels=("baseline",)),
            TimeseriesAnnotation(name="series"),
        )


def test_pandas_labeled_data_dataset_can_include_state_columns() -> None:
    labeled = make_pandas_labeled(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]}),
        segments=[
            SegmentInfo(
                segment_num=0,
                segment_start=0,
                segment_end=1,
                state=StateDescriptor(label="baseline", phase="warmup"),
            ),
            SegmentInfo(
                segment_num=1,
                segment_start=2,
                segment_end=3,
                state=StateDescriptor(label="shift"),
            ),
        ],
    )

    dataset = labeled.dataset(state_columns={"label": "label", "phase": "phase"})

    assert dataset["label"].tolist() == ["baseline", "baseline", "shift", "shift"]
    assert dataset["phase"].iloc[:2].tolist() == ["warmup", "warmup"]
    assert pd.isna(dataset["phase"].iloc[2])
    assert pd.isna(dataset["phase"].iloc[3])


def test_pandas_labeled_data_builds_state_series_with_missing_values() -> None:
    labeled = make_pandas_labeled(
        pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]}),
        segments=[
            SegmentInfo(
                segment_num=0,
                segment_start=0,
                segment_end=1,
                state=StateDescriptor(label="baseline", phase="warmup"),
            ),
            SegmentInfo(
                segment_num=1,
                segment_start=2,
                segment_end=2,
                state=StateDescriptor(label="shift"),
            ),
        ],
    )

    series = labeled._make_series_from_state_var("phase")

    assert series.iloc[:2].tolist() == ["warmup", "warmup"]
    assert pd.isna(series.iloc[2])


def test_labeled_data_merge_with_custom_annotation_builder() -> None:
    provider = make_univariate_labeled()
    other = make_univariate_labeled(data=(10.0, 11.0, 12.0, 20.0, 21.0, 22.0), name="other")
    custom_annotation = TimeseriesAnnotation(name="custom-merged")

    merged = type(provider).merge(
        [provider, other],
        annotation_builder=lambda _: custom_annotation,
    )

    assert merged.annotation is custom_annotation
