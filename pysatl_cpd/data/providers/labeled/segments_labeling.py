# -*- coding: ascii -*-
"""Segment labeling utilities for labeled time series providers."""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import Self, overload

import pandas as pd

from pysatl_cpd.data.typedefs import (
    BisegmentFilter,
    BisegmentInfo,
    SegmentFilter,
    SegmentInfo,
    StateDescriptor,
    StateValue,
    TransitionDescriptor,
)


class SegmentsLabeling(Sequence[SegmentInfo]):
    """
    Labeled segments with query and transformation capabilities.

    This class provides a sequence of segment information objects
    with methods for querying segments and bisegments, cutting
    slices, merging labelings, and converting to/from DataFrames.

    Parameters
    ----------
    segments
        Iterable of segment information objects.
    """

    def __init__(self, segments: Iterable[SegmentInfo]) -> None:
        self.__segment_info = tuple(segments)
        self._validate_segments()

    def _validate_segments(self) -> None:
        """Validate segment ordering and continuity invariants.

        Raises
        ------
        ValueError
            If segments are not ordered, contiguous, or consistently
            numbered.
        """
        if not self.__segment_info:
            return
        offset = self.__segment_info[0].segment_num
        for expected_num, segment in enumerate(self.__segment_info):
            if segment.segment_num != expected_num + offset:
                raise ValueError("Segments labeling must use consecutive segment numbers")
            if expected_num == 0:
                continue
            previous = self.__segment_info[expected_num - 1]
            if segment.segment_start <= previous.segment_end:
                raise ValueError("Segments labeling must not contain overlapping segments")
            if segment.segment_start != previous.segment_end + 1:
                raise ValueError("Segments labeling must be contiguous")

    def query_segments(self, filter_fn: SegmentFilter | None = None) -> Sequence[SegmentInfo]:
        """
        Query segments by optional filter.

        Parameters
        ----------
        filter_fn
            Optional predicate function for segment filtering.

        Returns
        -------
        segments
            Filtered sequence of segment information.
        """
        filter_fn = filter_fn if filter_fn is not None else (lambda _: True)
        return list(filter(filter_fn, self.__segment_info))

    def query_bisegments(self, filter_fn: BisegmentFilter | None = None) -> Sequence[BisegmentInfo]:
        """
        Query bisegments by optional filter.

        Parameters
        ----------
        filter_fn
            Optional predicate function for bisegment filtering.

        Returns
        -------
        bisegments
            Filtered sequence of bisegment information.
        """
        filter_fn = filter_fn if filter_fn is not None else (lambda _: True)
        return list(filter(filter_fn, map(BisegmentInfo.from_segment_tuple, pairwise(self.__segment_info))))

    @property
    def states(self) -> set[StateDescriptor]:
        """Return the set of states represented by the labeling.

        Returns
        -------
        states
            Set of unique state descriptors in this labeling.
        """
        return {_info.state for _info in self.__segment_info}

    @property
    def transitions(self) -> set[TransitionDescriptor]:
        """Return transitions between consecutive segments.

        Returns
        -------
        transitions
            Set of unique transition descriptors between consecutive
            segments.
        """
        return {
            TransitionDescriptor(curr_state=_curr.state, next_state=_next.state)
            for _curr, _next in pairwise(self.__segment_info)
        }

    def cut(self, start: int, stop: int) -> "SegmentsLabeling":
        """
        Cut a slice from the labeling.

        Parameters
        ----------
        start
            Start index of the slice.
        stop
            End index of the slice.

        Returns
        -------
        labeling
            New labeling containing the cut portion.

        Raises
        ------
        ValueError
            If start is negative, stop is less than start, or stop
            exceeds the data length.
        """
        if start < 0:
            raise ValueError("Slice start index must be non-negative")
        if stop < start:
            raise ValueError("Slice stop index must be greater than or equal to start index")
        if self.__segment_info and stop >= self.data_len:
            raise ValueError(f"Slice stop index {stop} exceeds data length {self.data_len}")
        segments = [
            segment for segment in self.__segment_info if segment.segment_end >= start and segment.segment_start <= stop
        ]
        return SegmentsLabeling(
            [
                SegmentInfo(
                    segment_num=index,
                    segment_start=max(segment.segment_start, start) - start,
                    segment_end=min(segment.segment_end, stop) - start,
                    state=segment.state,
                )
                for index, segment in enumerate(segments)
            ]
        )

    @classmethod
    def merge(cls, labelings: "Sequence[SegmentsLabeling]") -> "SegmentsLabeling":
        """
        Merge multiple labelings into one.

        Parameters
        ----------
        labelings
            Sequence of labelings to merge.

        Returns
        -------
        labeling
            Single merged labeling with adjusted offsets.
        """
        data_offset = 0
        segment_offset = 0
        merged_segment_info: list[SegmentInfo] = []
        for labeling in labelings:
            segments = list(labeling)
            merged_segment_info.extend(
                SegmentInfo(
                    segment_num=segment.segment_num + segment_offset,
                    segment_start=segment.segment_start + data_offset,
                    segment_end=segment.segment_end + data_offset,
                    state=segment.state,
                )
                for segment in segments
            )
            data_offset += labeling.data_len
            segment_offset += len(labeling)
        return SegmentsLabeling(merged_segment_info)

    @overload
    def __getitem__(self, index: int) -> SegmentInfo: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: int | slice) -> SegmentInfo | Self:
        """Return one segment or a sliced labeling.

        Parameters
        ----------
        index
            Integer index or slice.

        Returns
        -------
        item
            Segment information or a new labeling slice.
        """
        if isinstance(index, slice):
            return type(self)(self.__segment_info[index])
        return self.__segment_info[index]

    def __len__(self) -> int:
        """Return the number of segments.

        Returns
        -------
        length
            Number of labeled segments.
        """
        return len(self.__segment_info)

    @property
    def data_len(self) -> int:
        """Return the total covered data length.

        Returns
        -------
        length
            Total data length across all segments.
        """
        return self[-1].segment_end - self[0].segment_start + 1

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        segment_num_col: str | None = "segment",
        segment_start_col: str = "start",
        segment_end_col: str = "end",
    ) -> Self:
        """
        Create labeling from a DataFrame.

        Parameters
        ----------
        frame
            Input DataFrame containing segment data.
        segment_num_col
            Name of the column containing segment numbers.
        segment_start_col
            Name of the column containing segment start indices.
        segment_end_col
            Name of the column containing segment end indices.

        Returns
        -------
        labeling
            New labeling instance constructed from the DataFrame.
        """
        segment_info_seq: list[SegmentInfo] = []
        for idx, row in frame.iterrows():
            segment_start = row[segment_start_col]
            segment_end = row[segment_end_col]
            segment_num = row[segment_num_col] if segment_num_col else idx
            state_vars = row.drop([segment_start_col, segment_end_col])
            if segment_num_col is not None:
                state_vars = state_vars.drop(segment_num_col)
            state = StateDescriptor(**dict(state_vars))
            segment_info_seq.append(
                SegmentInfo(segment_num=segment_num, segment_start=segment_start, segment_end=segment_end, state=state)
            )
        return cls(segment_info_seq)

    def to_dataframe(
        self,
        segment_num_col: str | None = "segment",
        segment_start_col: str = "start",
        segment_end_col: str = "end",
    ) -> pd.DataFrame:
        """
        Convert labeling to a DataFrame.

        Parameters
        ----------
        segment_num_col
            Name of the column for segment numbers.
        segment_start_col
            Name of the column for segment start indices.
        segment_end_col
            Name of the column for segment end indices.

        Returns
        -------
        frame
            DataFrame representation of the labeling.
        """
        rows: list[dict[str, StateValue]] = []
        for info in self.__segment_info:
            row_dict: dict[str, StateValue] = {}
            if segment_num_col is not None:
                row_dict.update({segment_num_col: info.segment_num})
            row_dict.update({segment_start_col: info.segment_start, segment_end_col: info.segment_end})
            row_dict.update(dict(info.state))
            rows.append(row_dict)

        frame = pd.DataFrame(rows)
        frame[segment_start_col] = frame[segment_start_col].astype("int")
        frame[segment_end_col] = frame[segment_end_col].astype("int")
        if segment_num_col is not None:
            frame[segment_num_col] = frame[segment_num_col].astype("int")

        return frame
