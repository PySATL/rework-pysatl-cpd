# -*- coding: ascii -*-
"""
Tests for dataset contract.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.data import IDataset, StateDataset


class IDatasetContract:
    """Reusable checks for IDataset implementations."""

    @pytest.fixture
    def dataset(self):
        raise NotImplementedError

    def test_len_matches_number_of_timeseries(self, dataset: IDataset) -> None:
        assert len(dataset) == len(dataset.timeseries)

    def test_iteration_preserves_order(self, dataset: IDataset) -> None:
        assert list(dataset) == dataset.timeseries

    def test_getitem_returns_expected_provider(self, dataset: IDataset) -> None:
        assert dataset[0] == dataset.timeseries[0]

    def test_timeseries_returns_copy(self, dataset: IDataset) -> None:
        copy = dataset.timeseries
        assert copy == list(dataset)
        assert copy is not dataset.timeseries

    def test_states_union_matches_children(self, dataset: IDataset) -> None:
        assert dataset.states == set.union(*(provider.states for provider in dataset))

    def test_transitions_union_matches_children(self, dataset: IDataset) -> None:
        assert dataset.transitions == set.union(*(provider.transitions for provider in dataset))

    def test_empty_dataset_has_empty_states(self, dataset: IDataset) -> None:
        empty = type(dataset)([]) if not isinstance(dataset, StateDataset) else type(dataset)([], state=dataset.state)
        assert empty.states == set()

    def test_empty_dataset_has_empty_transitions(self, dataset: IDataset) -> None:
        empty = type(dataset)([]) if not isinstance(dataset, StateDataset) else type(dataset)([], state=dataset.state)
        assert empty.transitions == set()

    def test_train_test_split_returns_same_concrete_type(self, dataset: IDataset) -> None:
        train, test = dataset.train_test_split(0.5, random_state=7)
        assert type(train) is type(dataset)
        assert type(test) is type(dataset)

    def test_train_test_split_sizes_sum_to_original(self, dataset: IDataset) -> None:
        train, test = dataset.train_test_split(0.5, random_state=7)
        assert len(train) + len(test) == len(dataset)

    def test_train_test_split_seeded_is_deterministic(self, dataset: IDataset) -> None:
        first = dataset.train_test_split(0.5, random_state=7)
        second = dataset.train_test_split(0.5, random_state=7)
        assert first[0].timeseries == second[0].timeseries
        assert first[1].timeseries == second[1].timeseries

    def test_train_test_split_rejects_invalid_test_size(self, dataset: IDataset) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            dataset.train_test_split(1.5)

    def test_merge_non_empty_returns_labeled_provider(self, dataset: IDataset) -> None:
        merged = dataset.merge()
        assert len(merged) == sum(len(provider) for provider in dataset)

    def test_merge_empty_raises(self, dataset: IDataset) -> None:
        empty = type(dataset)([]) if not isinstance(dataset, StateDataset) else type(dataset)([], state=dataset.state)
        with pytest.raises(ValueError, match="empty dataset"):
            empty.merge()
