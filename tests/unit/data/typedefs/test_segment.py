# -*- coding: ascii -*-
"""
Tests for segment.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.data.typedefs import (
    BisegmentInfo,
    SegmentInfo,
    StateDescriptor,
    StateValue,
    TransitionDescriptor,
)
from pysatl_cpd.typedefs import stable_hash


def test_state_descriptor_mapping_behaviors_repr_and_hash() -> None:
    values: dict[str, StateValue] = {"beta": 2, "alpha": "baseline"}
    state = StateDescriptor(**values)

    assert state["alpha"] == "baseline"
    assert tuple(state) == ("beta", "alpha")
    assert len(state) == 2
    assert repr(state) == "alpha='baseline'|beta=2"
    assert hash(state) == stable_hash(state.data)


def test_transition_descriptor_repr_and_hash() -> None:
    current = StateDescriptor(label="baseline")
    nxt = StateDescriptor(label="shift")
    transition = TransitionDescriptor(curr_state=current, next_state=nxt)

    assert repr(transition) == "label='baseline'->label='shift'"
    assert hash(transition) == stable_hash(
        (
            transition.__class__.__module__,
            transition.__class__.__qualname__,
            current,
            nxt,
        )
    )


@pytest.mark.parametrize(
    ("segment_start", "segment_end", "message"),
    [
        (-1, 0, "Segment start index must be non-negative"),
        (3, 2, "Segment end index must be greater than or equal to segment start index"),
    ],
)
def test_segment_info_post_init_validates_bounds(
    segment_start: int,
    segment_end: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SegmentInfo(
            segment_num=0,
            segment_start=segment_start,
            segment_end=segment_end,
            state=StateDescriptor(label="baseline"),
        )


def test_segment_info_hash_uses_stable_hash() -> None:
    state = StateDescriptor(label="baseline")
    segment = SegmentInfo(segment_num=1, segment_start=2, segment_end=5, state=state)

    assert hash(segment) == stable_hash((segment.__class__.__module__, segment.__class__.__qualname__, 1, 2, 5, state))


def test_segment_info_from_change_points_and_states_rejects_invalid_counts() -> None:
    with pytest.raises(
        ValueError,
        match="Number of states must be exactly one more than number of change points",
    ):
        SegmentInfo.from_change_points_and_states(
            total_len=10,
            change_points=(3, 7),
            segment_states=(StateDescriptor(label="only-one-state"),),
        )


def test_segment_info_from_change_points_and_states_builds_segments() -> None:
    states = (
        StateDescriptor(label="baseline"),
        StateDescriptor(label="warning"),
        StateDescriptor(label="critical"),
    )

    segments = SegmentInfo.from_change_points_and_states(
        total_len=10,
        change_points=(3, 7),
        segment_states=states,
    )

    assert segments == (
        SegmentInfo(segment_num=0, segment_start=0, segment_end=2, state=states[0]),
        SegmentInfo(segment_num=1, segment_start=3, segment_end=6, state=states[1]),
        SegmentInfo(segment_num=2, segment_start=7, segment_end=9, state=states[2]),
    )


@pytest.mark.parametrize(
    ("bisegment_start", "bisegment_end", "change_point", "message"),
    [
        (-1, 2, 0, "Bisegment start index must be non-negative"),
        (3, 2, 3, "Bisegment end index must be greater than or equal to bisegment start index"),
        (3, 6, 2, "Change point must lie within the bisegment boundaries"),
    ],
)
def test_bisegment_info_post_init_validates_bounds(
    bisegment_start: int,
    bisegment_end: int,
    change_point: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BisegmentInfo(
            bisegment_num=0,
            bisegment_start=bisegment_start,
            bisegment_end=bisegment_end,
            change_point=change_point,
            transition=TransitionDescriptor(
                curr_state=StateDescriptor(label="baseline"),
                next_state=StateDescriptor(label="shift"),
            ),
        )


def test_bisegment_info_from_segment_tuple_and_hash() -> None:
    first = SegmentInfo(
        segment_num=4,
        segment_start=10,
        segment_end=14,
        state=StateDescriptor(label="baseline"),
    )
    second = SegmentInfo(
        segment_num=5,
        segment_start=15,
        segment_end=22,
        state=StateDescriptor(label="shift"),
    )

    bisegment = BisegmentInfo.from_segment_tuple((first, second))

    assert bisegment == BisegmentInfo(
        bisegment_num=4,
        bisegment_start=10,
        bisegment_end=22,
        change_point=15,
        transition=TransitionDescriptor(curr_state=first.state, next_state=second.state),
    )
    assert hash(bisegment) == stable_hash(
        (
            bisegment.__class__.__module__,
            bisegment.__class__.__qualname__,
            4,
            10,
            22,
            15,
            bisegment.transition,
        )
    )
