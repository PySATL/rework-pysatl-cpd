# -*- coding: ascii -*-
"""
Tests for labeled data contract applications.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pandas as pd
import pytest

from pysatl_cpd.data import PandasLabeledData, PlainMultivariateLabeledData, PlainUnivariateLabeledData
from pysatl_cpd.data.typedefs import SegmentInfo, StateDescriptor, TimeseriesAnnotation
from tests.contracts.data.test_labeled_data_contract import LabeledDataContract
from tests.support.providers import make_multivariate_labeled, make_pandas_labeled, make_univariate_labeled


class TestPlainUnivariateLabeledDataContract(LabeledDataContract):
    @pytest.fixture
    def provider(self) -> PlainUnivariateLabeledData[TimeseriesAnnotation]:
        return make_univariate_labeled()

    @pytest.fixture
    def other_provider(self) -> PlainUnivariateLabeledData[TimeseriesAnnotation]:
        return make_univariate_labeled((20.0, 21.0, 22.0, 30.0, 31.0, 32.0), name="other")


class TestPlainMultivariateLabeledDataContract(LabeledDataContract):
    @pytest.fixture
    def provider(self) -> PlainMultivariateLabeledData[TimeseriesAnnotation]:
        return make_multivariate_labeled()

    @pytest.fixture
    def other_provider(self) -> PlainMultivariateLabeledData[TimeseriesAnnotation]:
        return make_multivariate_labeled(
            ((20.0, 200.0), (21.0, 210.0), (22.0, 220.0), (30.0, 300.0), (31.0, 310.0), (32.0, 320.0)),
            name="other",
        )


class TestPandasLabeledDataContract(LabeledDataContract):
    @pytest.fixture
    def provider(self) -> PandasLabeledData[TimeseriesAnnotation]:
        return make_pandas_labeled()

    @pytest.fixture
    def other_provider(self) -> PandasLabeledData[TimeseriesAnnotation]:
        return make_pandas_labeled(
            pd.DataFrame(
                {
                    "x": [20.0, 21.0, 22.0, 30.0, 31.0, 32.0],
                    "y": [200.0, 210.0, 220.0, 300.0, 310.0, 320.0],
                }
            ),
            name="other",
        )


def test_labeled_data_single_segment_has_no_change_points() -> None:
    provider = make_univariate_labeled(
        data=(1.0, 2.0, 3.0),
        segments=[SegmentInfo(segment_num=0, segment_start=0, segment_end=2, state=StateDescriptor(label="steady"))],
    )
    assert provider.change_points == ()
    assert provider.query_bisegments() == []


def test_pandas_labeled_data_dataset_contains_segment_columns() -> None:
    provider = make_pandas_labeled()
    dataset = provider.dataset()
    assert list(dataset.columns) == ["x", "y", "segment", "start", "end"]


def test_labeled_data_cut_uses_explicit_annotation() -> None:
    provider = make_univariate_labeled()
    annotation = TimeseriesAnnotation(name="custom-cut")
    assert provider.cut(1, 3, annotation=annotation).annotation is annotation
