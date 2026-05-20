# -*- coding: ascii -*-
"""Base dataset sequence abstractions for labeled time series."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import random
from collections.abc import Iterator, MutableSequence, Sequence
from typing import Self, cast

from pysatl_cpd.data.providers.labeled import LabeledData
from pysatl_cpd.data.typedefs import StateDescriptor, TimeseriesAnnotation, TransitionDescriptor


class IDataset[
    DataT,
    AnnotationT: TimeseriesAnnotation,
](Sequence[LabeledData[DataT, AnnotationT]]):
    """Common sequence interface for labeled time series collections.

    Parameters
    ----------
    timeseries
        Sequence of labeled data instances.
    """

    def __init__(self, timeseries: Sequence[LabeledData[DataT, AnnotationT]]) -> None:
        self._timeseries = list(timeseries)

    def __getitem__(self, index: int) -> LabeledData[DataT, AnnotationT]:  # type: ignore[override]
        """Return one labeled time series by index.

        Parameters
        ----------
        index
            Position of the item to return.

        Returns
        -------
        timeseries
            Labeled time series stored at the given position.
        """
        return self._timeseries[index]

    def __iter__(self) -> Iterator[LabeledData[DataT, AnnotationT]]:
        """Iterate over stored labeled time series.

        Returns
        -------
        iterator
            Iterator over stored labeled providers.
        """
        return iter(self._timeseries)

    def __len__(self) -> int:
        """Return the dataset size.

        Returns
        -------
        length
            Number of labeled time series in the dataset.
        """
        return len(self._timeseries)

    @property
    def timeseries(self) -> MutableSequence[LabeledData[DataT, AnnotationT]]:
        """Return a copy of the stored labeled providers.

        Returns
        -------
        timeseries
            Copy of the internal list of labeled data instances.
        """
        return list(self._timeseries)

    @property
    def states(self) -> set[StateDescriptor]:
        """Return the union of all states from the dataset.

        Returns
        -------
        states
            Set of all distinct state descriptors across stored series.
        """
        return set.union(*(ts.states for ts in self._timeseries)) if self._timeseries else set()

    @property
    def transitions(self) -> set[TransitionDescriptor]:
        """Return the union of all transitions from the dataset.

        Returns
        -------
        transitions
            Set of all distinct transition descriptors across stored series.
        """
        return set.union(*(ts.transitions for ts in self._timeseries)) if self._timeseries else set()

    def train_test_split(self, test_size: float, random_state: int | None = None) -> tuple[Self, Self]:
        """Split the dataset into train and test subsets.

        Parameters
        ----------
        test_size
            Fraction of items to place into the test split.
        random_state
            Optional random seed for reproducible shuffling.

        Returns
        -------
        splits
            Train and test datasets of the same concrete type.

        Raises
        ------
        ValueError
            If test_size is not between 0 and 1.
        """
        if not 0.0 <= test_size <= 1.0:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

        n_test = int(len(self._timeseries) * test_size)

        if random_state is not None:
            rng = random.Random(random_state)
            indices = list(range(len(self._timeseries)))
            rng.shuffle(indices)
            test_indices = set(indices[:n_test])
        else:
            test_indices = set(range(n_test))

        train_ts = [ts for i, ts in enumerate(self._timeseries) if i not in test_indices]
        test_ts = [ts for i, ts in enumerate(self._timeseries) if i in test_indices]

        return self._build_like(train_ts), self._build_like(test_ts)

    def _build_like(self, timeseries: Sequence[LabeledData[DataT, AnnotationT]]) -> Self:
        """
        Build a new instance with the same type and given timeseries.

        Parameters
        ----------
        timeseries
            Sequence of labeled data instances.

        Returns
        -------
        dataset
            New instance of the same type with given timeseries.
        """
        return type(self)(timeseries)

    def merge(self) -> LabeledData[DataT, TimeseriesAnnotation]:
        """Merge all stored providers into a single labeled provider.

        Returns
        -------
        provider
            Labeled provider containing all time series in sequence.

        Raises
        ------
        ValueError
            If the dataset is empty.
        """
        if not self._timeseries:
            raise ValueError("Cannot merge empty dataset")

        first = self._timeseries[0]
        return cast(LabeledData[DataT, TimeseriesAnnotation], type(first).merge(self._timeseries))
