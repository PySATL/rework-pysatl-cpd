# -*- coding: ascii -*-
"""Segment and state descriptors for the data layer."""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from pysatl_cpd.typedefs import frozendict, stable_hash

type StateValue = str | int | float | bool


class StateDescriptor(Mapping[str, StateValue]):
    """
    Immutable mapping for state attributes.

    This class provides an immutable mapping interface for storing
    state attributes as key-value pairs. It uses frozendict
    internally to ensure hashability and immutability.

    Parameters
    ----------
    **kwargs
        Key-value pairs representing state attributes.
    """

    def __init__(self, **kwargs: StateValue):
        self.data: frozendict[str, StateValue] = frozendict(**kwargs)

    def __getitem__(self, key: str) -> StateValue:
        """
        Get state value by key.

        Parameters
        ----------
        key
            The key to look up.

        Returns
        -------
        value
            The value associated with the key.
        """
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over state keys.

        Returns
        -------
        iterator
            Iterator over the state keys.
        """
        return iter(self.data)

    def __len__(self) -> int:
        """
        Get number of state attributes.

        Returns
        -------
        count
            Number of key-value pairs in the state.
        """
        return len(self.data)

    # def get(self, key: str, default: object = None) -> StateValue | object:
    #    return self.data.get(key, default)
    def __repr__(self) -> str:
        """Return a string representation of the state descriptor.

        Returns
        -------
        str
            Pipe-separated key=value pairs sorted by key.
        """
        return "|".join(f"{key}={value!r}" for key, value in sorted(self.data.items()))

    def __hash__(self) -> int:
        """
        Get hash of the state descriptor.

        Returns
        -------
        hash
            Hash value of the underlying frozendict.
        """
        return stable_hash(self.data)


@dataclass(frozen=True, kw_only=True, slots=True)
class TransitionDescriptor:
    """
    Transition between two states.

    This class represents a transition from a current state to a
    next state, used for tracking state changes in time series
    segments.

    Attributes
    ----------
    curr_state
        The current state before the transition.
    next_state
        The next state after the transition.
    """

    curr_state: StateDescriptor
    next_state: StateDescriptor

    def __repr__(self) -> str:
        """Return a string representation of the transition.

        Returns
        -------
        str
            Current state and next state separated by an arrow.
        """
        return f"{self.curr_state!r}->{self.next_state!r}"

    def __hash__(self) -> int:
        """Return a stable hash for the transition descriptor."""
        return stable_hash((type(self).__module__, type(self).__qualname__, self.curr_state, self.next_state))


@dataclass(frozen=True, kw_only=True, slots=True)
class SegmentInfo:
    """
    Information about a time series segment.

    This class represents a contiguous segment of a time series
    with a specific state, including the segment number and
    start/end indices.

    Attributes
    ----------
    segment_num
        Sequential number of the segment.
    segment_start
        Start index of the segment (inclusive, zero-based).
    segment_end
        End index of the segment (inclusive, zero-based).
    state
        The state descriptor for this segment.
    """

    segment_num: int
    segment_start: int
    segment_end: int
    state: StateDescriptor

    def __post_init__(self) -> None:
        """Validate segment indices.

        Raises
        ------
        ValueError
            If segment_start is negative, or segment_end is less
            than segment_start.
        """
        if self.segment_start < 0:
            raise ValueError("Segment start index must be non-negative")
        if self.segment_end < self.segment_start:
            raise ValueError("Segment end index must be greater than or equal to segment start index")

    def __hash__(self) -> int:
        """Return a stable hash for segment metadata."""
        return stable_hash(
            (
                type(self).__module__,
                type(self).__qualname__,
                self.segment_num,
                self.segment_start,
                self.segment_end,
                self.state,
            )
        )

    @classmethod
    def from_change_points_and_states(
        cls,
        total_len: int,
        change_points: tuple[int, ...],
        segment_states: Sequence[StateDescriptor],
    ) -> tuple["SegmentInfo", ...]:
        """
        Create segment infos from change points and states.

        This class method constructs a tuple of SegmentInfo objects
        from the total length of the series, change point indices,
        and corresponding segment states.

        Parameters
        ----------
        total_len
            Total length of the time series.
        change_points
            Tuple of change point indices (zero-based).
        segment_states
            Sequence of state descriptors for each segment.

        Returns
        -------
        segments
            Tuple of SegmentInfo objects for each segment.

        Raises
        ------
        ValueError
            If the number of segment_states is not exactly one more
            than the number of change_points.
        """
        if len(change_points) + 1 != len(segment_states):
            raise ValueError("Number of states must be exactly one more than number of change points")
        segments: list[SegmentInfo] = []
        start = 0
        for i, state in enumerate(segment_states):
            end = change_points[i] - 1 if i < len(change_points) else total_len - 1
            if i > 0:
                start = change_points[i - 1]
            segments.append(
                cls(
                    segment_num=i,
                    segment_start=start,
                    segment_end=end,
                    state=state,
                )
            )
        return tuple(segments)


@dataclass(frozen=True, kw_only=True, slots=True)
class BisegmentInfo:
    """
    Information about a bisegment spanning a change point.

    This class represents a bisegment that spans a change point,
    including the transition between two states.

    Attributes
    ----------
    bisegment_num
        Sequential number of the bisegment.
    bisegment_start
        Start index of the bisegment (inclusive, zero-based).
    bisegment_end
        End index of the bisegment (inclusive, zero-based).
    change_point
        Index of the change point within the bisegment.
    transition
        Transition descriptor for the state change.
    """

    bisegment_num: int
    bisegment_start: int
    bisegment_end: int
    change_point: int
    transition: TransitionDescriptor

    def __post_init__(self) -> None:
        """Validate bisegment boundaries and change-point position.

        Raises
        ------
        ValueError
            If bisegment_start is negative, bisegment_end is less
            than bisegment_start, or change_point is not within
            [bisegment_start, bisegment_end].
        """
        if self.bisegment_start < 0:
            raise ValueError("Bisegment start index must be non-negative")
        if self.bisegment_end < self.bisegment_start:
            raise ValueError("Bisegment end index must be greater than or equal to bisegment start index")
        if not self.bisegment_start <= self.change_point <= self.bisegment_end:
            raise ValueError("Change point must lie within the bisegment boundaries")

    def __hash__(self) -> int:
        """Return a stable hash for bisegment metadata."""
        return stable_hash(
            (
                type(self).__module__,
                type(self).__qualname__,
                self.bisegment_num,
                self.bisegment_start,
                self.bisegment_end,
                self.change_point,
                self.transition,
            )
        )

    @classmethod
    def from_segment_tuple(cls, segments: tuple[SegmentInfo, SegmentInfo]) -> "BisegmentInfo":
        """
        Create bisegment info from a pair of segments.

        This method constructs a BisegmentInfo from two consecutive
        SegmentInfo objects, extracting the bisegment boundaries
        and creating a transition descriptor.

        Parameters
        ----------
        segments
            Tuple of two consecutive SegmentInfo objects.

        Returns
        -------
        bisegment
            The constructed BisegmentInfo object.
        """
        return BisegmentInfo(
            bisegment_num=segments[0].segment_num,
            bisegment_start=segments[0].segment_start,
            bisegment_end=segments[1].segment_end,
            change_point=segments[1].segment_start,
            transition=TransitionDescriptor(curr_state=segments[0].state, next_state=segments[1].state),
        )
