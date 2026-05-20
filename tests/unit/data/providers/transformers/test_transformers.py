# -*- coding: ascii -*-
"""
Tests for transformers.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pandas as pd
import pytest

from pysatl_cpd.data import NDArrayUnivariateProvider, PandasLabeledData, PlainUnivariateLabeledData
from pysatl_cpd.data.providers.plain.pd_provider import PandasDataProvider
from pysatl_cpd.data.providers.transformers import ColumnFeatureCreator, ColumnsSelectorTransformer
from pysatl_cpd.data.typedefs import SegmentInfo, StateDescriptor, TimeseriesAnnotation, UnlabeledTimeseriesAnnotation
from pysatl_cpd.typedefs import frozendict


def test_pandas_provider_create_feature_column_appends_column() -> None:
    provider = PandasDataProvider(
        pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}),
        UnlabeledTimeseriesAnnotation(name="series"),
    )

    result = provider.create_feature_column(name="sum", mapping=lambda row: row["x"] + row["y"])

    assert list(result.columns) == ["x", "y", "sum"]
    assert result.dataset["sum"].tolist() == [4.0, 6.0]
    assert result.name == "series"


def test_pandas_provider_select_columns_preserves_name_by_default() -> None:
    provider = PandasDataProvider(
        pd.DataFrame({"x": [1.0], "y": [2.0]}),
        UnlabeledTimeseriesAnnotation(name="series"),
    )

    result = provider.select_columns(["x"])

    assert result.name == "series"


def test_pandas_provider_select_columns_can_rename_provider() -> None:
    provider = PandasDataProvider(
        pd.DataFrame({"x": [1.0], "y": [2.0]}),
        UnlabeledTimeseriesAnnotation(name="series"),
    )

    result = provider.select_columns(["x"], rename_provider=True)

    assert result.name == "series[x]"


def test_pandas_provider_create_feature_column_rejects_duplicate_name() -> None:
    provider = PandasDataProvider(
        pd.DataFrame({"x": [1.0], "y": [2.0]}),
        UnlabeledTimeseriesAnnotation(name="series"),
    )

    with pytest.raises(ValueError, match="already exists"):
        provider.create_feature_column(name="x", mapping=lambda row: row["x"])


def test_pandas_provider_create_feature_column_can_rename_provider() -> None:
    provider = PandasDataProvider(
        pd.DataFrame({"x": [1.0], "y": [2.0]}),
        UnlabeledTimeseriesAnnotation(name="series"),
    )

    result = provider.create_feature_column(name="sum", mapping=lambda row: row["x"] + row["y"], rename_provider=True)

    assert result.name == "series[+sum]"


def test_pandas_labeled_data_create_feature_column_preserves_labeling() -> None:
    labeled = _make_labeled_provider()

    result = labeled.create_feature_column(name="sum", mapping=lambda row: row["x"] + row["y"])

    assert result.feature_columns == ["x", "y", "sum"]
    assert result.change_points == (2,)
    assert result.dataset()["sum"].tolist() == [11.0, 22.0, 33.0, 44.0]
    assert result.annotation.metadata == labeled.annotation.metadata
    assert result.name == labeled.name


def test_pandas_labeled_data_select_columns_can_rename_provider() -> None:
    labeled = _make_labeled_provider()

    result = labeled.select_columns(feature_columns=["x"], rename_provider=True)

    assert result.name == "series[x]"


def test_pandas_labeled_data_create_feature_column_can_rename_provider() -> None:
    labeled = _make_labeled_provider()

    result = labeled.create_feature_column(name="sum", mapping=lambda row: row["x"] + row["y"], rename_provider=True)

    assert result.name == "series[+sum]"


def test_column_feature_creator_supports_pandas_data_provider() -> None:
    provider = PandasDataProvider(
        pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}),
        UnlabeledTimeseriesAnnotation(name="series"),
    )

    result = ColumnFeatureCreator(name="mean_xy", mapping=lambda row: (row["x"] + row["y"]) / 2).transform(provider)

    assert list(result.columns) == ["x", "y", "mean_xy"]
    assert result.dataset["mean_xy"].tolist() == [2.0, 3.0]
    assert result.name == "series"


def test_column_feature_creator_can_rename_provider() -> None:
    provider = PandasDataProvider(
        pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}),
        UnlabeledTimeseriesAnnotation(name="series"),
    )

    result = ColumnFeatureCreator(
        name="mean_xy", mapping=lambda row: (row["x"] + row["y"]) / 2, rename_provider=True
    ).transform(provider)

    assert result.name == "series[+mean_xy]"


def test_column_feature_creator_supports_pandas_labeled_data() -> None:
    labeled = _make_labeled_provider()

    result = ColumnFeatureCreator(name="sum", mapping=lambda row: row["x"] + row["y"]).transform(labeled)

    assert result.feature_columns == ["x", "y", "sum"]
    assert result.change_points == (2,)
    assert result.name == labeled.name


def test_column_feature_creator_rejects_unsupported_provider() -> None:
    provider = PlainUnivariateLabeledData.from_unlabeled_data(
        NDArrayUnivariateProvider(np.array([1.0]), UnlabeledTimeseriesAnnotation(name="unsupported")),
        [SegmentInfo(segment_num=0, segment_start=0, segment_end=0, state=StateDescriptor(label="a"))],
        TimeseriesAnnotation(name="unsupported"),
    )

    with pytest.raises(TypeError, match="only supports PandasDataProvider and PandasLabeledData"):
        ColumnFeatureCreator(name="sum", mapping=lambda row: row["x"]).transform(provider)


def test_column_feature_creator_composes_with_columns_selector() -> None:
    labeled = _make_labeled_provider()
    transformer = ColumnFeatureCreator(
        name="sum", mapping=lambda row: row["x"] + row["y"]
    ) & ColumnsSelectorTransformer(["sum"])

    result = transformer.transform(labeled)

    assert result.feature_columns == ["sum"]
    assert list(result.dataset()["sum"]) == [11.0, 22.0, 33.0, 44.0]


def test_transformer_hash_is_stable_for_same_annotation() -> None:
    left = ColumnsSelectorTransformer(["x"], rename_provider=True)
    right = ColumnsSelectorTransformer(["x"], rename_provider=True)

    assert hash(left) == hash(right)


def test_transformer_hash_changes_with_annotation() -> None:
    left = ColumnsSelectorTransformer(["x"])
    right = ColumnsSelectorTransformer(["y"])

    assert hash(left) != hash(right)


def test_columns_selector_transformer_can_preserve_provider_name() -> None:
    labeled = _make_labeled_provider()

    result = ColumnsSelectorTransformer(["x"]).transform(labeled)

    assert result.name == labeled.name


def test_columns_selector_transformer_can_rename_provider() -> None:
    labeled = _make_labeled_provider()

    result = ColumnsSelectorTransformer(["x"], rename_provider=True).transform(labeled)

    assert result.name == "series[x]"


def test_columns_selector_transformer_rejects_empty_columns() -> None:
    labeled = _make_labeled_provider()

    with pytest.raises(ValueError, match="At least one column must be selected"):
        ColumnsSelectorTransformer([]).transform(labeled)


def test_columns_selector_transformer_rejects_non_string_columns() -> None:
    labeled = _make_labeled_provider()

    with pytest.raises(ValueError, match="only supports column names"):
        ColumnsSelectorTransformer(["x", 1]).transform(labeled)


def test_columns_selector_transformer_rejects_non_pandas_labeled_data() -> None:
    provider = PandasDataProvider(
        pd.DataFrame({"x": [1.0], "y": [2.0]}),
        UnlabeledTimeseriesAnnotation(name="series"),
    )

    with pytest.raises(TypeError, match="only supports PandasLabeledData providers"):
        ColumnsSelectorTransformer(["x"]).transform(provider)


def test_column_feature_creator_rejects_empty_name() -> None:
    labeled = _make_labeled_provider()

    with pytest.raises(ValueError, match="must be non-empty"):
        ColumnFeatureCreator(name="", mapping=lambda row: row["x"]).transform(labeled)


def test_column_feature_creator_rejects_non_callable_mapping() -> None:
    labeled = _make_labeled_provider()

    with pytest.raises(TypeError, match="mapping must be callable"):
        ColumnFeatureCreator(name="sum", mapping="not-callable").transform(labeled)


def _make_labeled_provider() -> PandasLabeledData[TimeseriesAnnotation]:
    unlabeled = PandasDataProvider(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]}),
        UnlabeledTimeseriesAnnotation(name="series", metadata=frozendict.from_mapping({"source": "test"})),
    )
    return PandasLabeledData.from_unlabeled_data(
        unlabeled,
        [
            SegmentInfo(segment_num=0, segment_start=0, segment_end=1, state=StateDescriptor(label="a")),
            SegmentInfo(segment_num=1, segment_start=2, segment_end=3, state=StateDescriptor(label="b")),
        ],
        TimeseriesAnnotation(name="series", metadata=frozendict.from_mapping({"kind": "demo"})),
    )
