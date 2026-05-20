# -*- coding: ascii -*-
"""Datasets of fixed-state labeled time series."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any, Self, cast

from pysatl_cpd.data.dataset.dataset import Dataset
from pysatl_cpd.data.dataset.idataset import IDataset
from pysatl_cpd.data.providers.labeled import LabeledData
from pysatl_cpd.data.typedefs import NoChangeSeriesAnnotation, ProviderType, SegmentInfo, StateDescriptor
from pysatl_cpd.typedefs import frozendict


class StateDataset[DataT](IDataset[DataT, NoChangeSeriesAnnotation]):
    """Dataset of fixed-state series without change points.

    All series in this dataset share a common state and have no
    change points within them. Typically created from a larger
    Dataset by slicing fixed-state segments.

    Parameters
    ----------
    timeseries
        Sequence of labeled data instances sharing the same state.
    state
        Optional explicit state descriptor; derived from the first
        timeseries if not provided.
    """

    def __init__(
        self,
        timeseries: Sequence[LabeledData[DataT, NoChangeSeriesAnnotation]],
        *,
        state: StateDescriptor | None = None,
    ) -> None:
        super().__init__(timeseries)
        self._state = self._resolve_state(state)

    @property
    def state(self) -> StateDescriptor:
        """
        Return the fixed state descriptor for this dataset.

        Returns
        -------
        state
            State descriptor for all timeseries in the dataset.
        """
        return self._state

    def _build_like(self, timeseries: Sequence[LabeledData[DataT, NoChangeSeriesAnnotation]]) -> Self:
        """
        Build a new instance with the same type and state.

        Parameters
        ----------
        timeseries
            Sequence of labeled data instances.

        Returns
        -------
        dataset
            New instance with same state.
        """
        return type(self)(timeseries, state=self._state)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset[DataT, Any],
        slice_length: int,
        *,
        state: StateDescriptor,
        keep_remainder: bool = False,
    ) -> StateDataset[DataT]:
        """
        Create a StateDataset from a Dataset by filtering and slicing.

        Parameters
        ----------
        dataset
            Source dataset to extract state segments from.
        slice_length
            Length of each slice to create.
        state
            State to filter segments by.
        keep_remainder
            Whether to keep remainder as final slice.

        Returns
        -------
        dataset
            New StateDataset with sliced timeseries.

        Raises
        ------
        ValueError
            If slice_length is not positive, or no segments are found
            for the given state.
        """
        if slice_length <= 0:
            raise ValueError(f"slice_length must be positive, got {slice_length}")

        segments_dataset = dataset.filter_by_segments(lambda segment: segment.state == state)
        if not segments_dataset.timeseries:
            raise ValueError(f"No segments found for state {state}")

        merged_provider = segments_dataset.merge()
        merged_len = len(merged_provider)
        state_timeseries: list[LabeledData[DataT, NoChangeSeriesAnnotation]] = []

        for start, stop in cls._slice_bounds(merged_len, slice_length, keep_remainder):
            metadata = frozendict.from_mapping(
                {
                    **dict(merged_provider.annotation.metadata),
                    "state_window_start": start,
                    "state_window_stop": stop,
                }
            )
            annotation = NoChangeSeriesAnnotation(
                name=f"{merged_provider.name}[state {start}:{stop}]",
                source=merged_provider.annotation.source,
                state=state,
                metadata=metadata,
            )
            sliced_unlabeled = merged_provider.unlabeled.cut(start, stop)
            slice_labeling = [
                SegmentInfo(
                    segment_num=0,
                    segment_start=0,
                    segment_end=stop - start,
                    state=state,
                )
            ]
            state_timeseries.append(
                cast(
                    LabeledData[DataT, NoChangeSeriesAnnotation],
                    type(merged_provider)(sliced_unlabeled, slice_labeling, annotation),
                )
            )

        return cls(state_timeseries, state=state)

    @staticmethod
    def _slice_bounds(total_len: int, slice_length: int, keep_remainder: bool) -> list[tuple[int, int]]:
        """
        Compute slice bounds for splitting total length into chunks.

        Parameters
        ----------
        total_len
            Total length to slice.
        slice_length
            Length of each slice.
        keep_remainder
            Whether to keep remainder as final slice.

        Returns
        -------
        bounds
            List of (start, stop) tuples for slicing.
        """
        if total_len < slice_length:
            return [(0, total_len - 1)] if keep_remainder and total_len > 0 else []

        bounds = [(start, start + slice_length - 1) for start in range(0, total_len - slice_length + 1, slice_length)]
        remainder_start = len(bounds) * slice_length
        if keep_remainder and remainder_start < total_len:
            bounds.append((remainder_start, total_len - 1))
        return bounds

    def _resolve_state(self, state: StateDescriptor | None) -> StateDescriptor:
        """
        Resolve the state from explicit value or timeseries.

        Parameters
        ----------
        state
            Explicit state or None to derive from timeseries.

        Returns
        -------
        state
            Resolved state descriptor.

        Raises
        ------
        ValueError
            If state cannot be resolved.
        """
        if not self._timeseries:
            if state is None:
                raise ValueError("StateDataset must contain at least one timeseries or an explicit state")
            return state

        first_annotation = self._timeseries[0].annotation
        if first_annotation.provider_type is not ProviderType.NO_CHANGE:
            raise ValueError("StateDataset requires NoChangeSeriesAnnotation providers")

        dataset_state = first_annotation.state
        for provider in self._timeseries[1:]:
            if provider.annotation.provider_type is not ProviderType.NO_CHANGE:
                raise ValueError("StateDataset requires NoChangeSeriesAnnotation providers")
            if provider.annotation.state != dataset_state:
                raise ValueError("StateDataset requires all providers to share the same state")

        if state is not None and state != dataset_state:
            raise ValueError("Explicit state does not match providers state")

        return dataset_state
