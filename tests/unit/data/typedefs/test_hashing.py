# -*- coding: ascii -*-
"""
Tests for hashing.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np

from pysatl_cpd.data.typedefs import (
    BisegmentAnnotation,
    BisegmentInfo,
    NoChangeSeriesAnnotation,
    SegmentAnnotation,
    SegmentInfo,
    StateDescriptor,
    TimeseriesAnnotation,
    TransitionDescriptor,
    UnlabeledTimeseriesAnnotation,
    frozendict,
)
from pysatl_cpd.typedefs import stable_hash


def test_timeseries_annotation_hash_uses_stable_hash() -> None:
    annotation = TimeseriesAnnotation(
        name="series",
        source="source",
        metadata=frozendict.from_mapping({"fold": 1}),
    )

    assert hash(annotation) == stable_hash(
        (annotation.__class__.__module__, annotation.__class__.__qualname__, "series", "source", annotation.metadata)
    )


def test_annotation_subclasses_hash_their_value_fields() -> None:
    state = StateDescriptor(label="baseline")
    transition = TransitionDescriptor(curr_state=state, next_state=StateDescriptor(label="shift"))

    segment = SegmentAnnotation(name="segment", state=state)
    no_change = NoChangeSeriesAnnotation(name="steady", state=state)
    bisegment = BisegmentAnnotation(name="pair", transition=transition)
    unlabeled = UnlabeledTimeseriesAnnotation(name="raw")

    assert hash(segment) == stable_hash(
        (segment.__class__.__module__, segment.__class__.__qualname__, "segment", None, segment.metadata, state)
    )
    assert hash(no_change) == stable_hash(
        (no_change.__class__.__module__, no_change.__class__.__qualname__, "steady", None, no_change.metadata, state)
    )
    assert hash(bisegment) == stable_hash(
        (
            bisegment.__class__.__module__,
            bisegment.__class__.__qualname__,
            "pair",
            None,
            bisegment.metadata,
            transition,
        )
    )
    assert hash(unlabeled) == stable_hash(
        (unlabeled.__class__.__module__, unlabeled.__class__.__qualname__, "raw", None, unlabeled.metadata)
    )


def test_state_descriptor_hash_uses_stable_hash() -> None:
    state = StateDescriptor(label="baseline", score=1)

    assert hash(state) == stable_hash(state.data)


def test_transition_descriptor_hash_uses_stable_hash() -> None:
    transition = TransitionDescriptor(
        curr_state=StateDescriptor(label="baseline"),
        next_state=StateDescriptor(label="shift"),
    )

    assert hash(transition) == stable_hash(
        (
            transition.__class__.__module__,
            transition.__class__.__qualname__,
            transition.curr_state,
            transition.next_state,
        )
    )


def test_segment_info_hash_uses_stable_hash() -> None:
    state = StateDescriptor(label="baseline")
    segment = SegmentInfo(segment_num=1, segment_start=2, segment_end=5, state=state)

    assert hash(segment) == stable_hash((segment.__class__.__module__, segment.__class__.__qualname__, 1, 2, 5, state))


def test_bisegment_info_hash_uses_stable_hash() -> None:
    transition = TransitionDescriptor(
        curr_state=StateDescriptor(label="baseline"),
        next_state=StateDescriptor(label="shift"),
    )
    bisegment = BisegmentInfo(
        bisegment_num=0,
        bisegment_start=0,
        bisegment_end=9,
        change_point=4,
        transition=transition,
    )

    assert hash(bisegment) == stable_hash(
        (
            bisegment.__class__.__module__,
            bisegment.__class__.__qualname__,
            0,
            0,
            9,
            4,
            transition,
        )
    )


def test_stable_hash_supports_numpy_backed_state_values() -> None:
    value = np.array([1.0, 2.0, 3.0])

    assert stable_hash(value) == stable_hash(np.array([1.0, 2.0, 3.0]))
