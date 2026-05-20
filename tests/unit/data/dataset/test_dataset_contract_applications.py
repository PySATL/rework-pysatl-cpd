# -*- coding: ascii -*-
"""
Tests for dataset contract applications.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.data import Dataset, StateDataset
from pysatl_cpd.data.typedefs import NoChangeSeriesAnnotation
from tests.contracts.data.test_dataset_contract import IDatasetContract
from tests.support.datasets import make_state_dataset
from tests.support.providers import make_univariate_labeled


class TestDatasetContract(IDatasetContract):
    @pytest.fixture
    def dataset(self) -> Dataset[float, object]:
        return Dataset(
            [
                make_univariate_labeled(name="series-a"),
                make_univariate_labeled(data=(7.0, 8.0, 9.0, 10.0, 11.0, 12.0), name="series-b"),
            ]
        )


class TestStateDatasetContract(IDatasetContract):
    @pytest.fixture
    def dataset(self) -> StateDataset[float]:
        return make_state_dataset()


def test_dataset_filter_by_annotation_returns_same_type() -> None:
    dataset = Dataset([make_univariate_labeled(name="keep-a"), make_univariate_labeled(name="drop-b")])
    filtered = dataset.filter_by_annotation(lambda annotation: annotation.name.startswith("keep"))
    assert isinstance(filtered, Dataset)
    assert [provider.name for provider in filtered] == ["keep-a"]


def test_dataset_filter_by_segments_returns_segments_dataset() -> None:
    dataset = Dataset([make_univariate_labeled(name="segments")])
    filtered = dataset.filter_by_segments(lambda segment: segment.state["label"] == "baseline")
    assert len(filtered) == 1
    assert filtered[0].annotation.provider_type is not NoChangeSeriesAnnotation.provider_type  # type: ignore[comparison-overlap]


def test_dataset_filter_by_bisegments_returns_bisegments_dataset() -> None:
    dataset = Dataset([make_univariate_labeled(name="bisegments")])
    filtered = dataset.filter_by_bisegments(lambda bisegment: bisegment.transition.next_state["label"] == "shift")
    assert len(filtered) == 1
    assert filtered[0].annotation.provider_type.name == "BISEGMENT"
