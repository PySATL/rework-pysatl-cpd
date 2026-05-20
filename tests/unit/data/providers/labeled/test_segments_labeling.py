# -*- coding: ascii -*-
"""
Tests for segments labeling.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pandas as pd
import pytest

from pysatl_cpd.data.providers.labeled.segments_labeling import SegmentsLabeling
from pysatl_cpd.data.typedefs import BisegmentInfo, SegmentInfo, StateDescriptor, TransitionDescriptor


def _state(label: str, phase: int | None = None) -> StateDescriptor:
    kwargs = {"label": label}
    if phase is not None:
        kwargs["phase"] = phase
    return StateDescriptor(**kwargs)


def _segments() -> list[SegmentInfo]:
    return [
        SegmentInfo(segment_num=0, segment_start=0, segment_end=2, state=_state("a", 1)),
        SegmentInfo(segment_num=1, segment_start=3, segment_end=5, state=_state("b", 2)),
        SegmentInfo(segment_num=2, segment_start=6, segment_end=8, state=_state("a", 1)),
    ]


def test_empty_labeling_is_allowed() -> None:
    labeling = SegmentsLabeling([])

    assert len(labeling) == 0
    assert labeling.query_segments(None) == []
    assert labeling.query_bisegments(None) == []


def test_validate_segments_rejects_non_consecutive_segment_numbers() -> None:
    with pytest.raises(ValueError, match="consecutive segment numbers"):
        SegmentsLabeling(
            [
                SegmentInfo(segment_num=0, segment_start=0, segment_end=1, state=_state("a")),
                SegmentInfo(segment_num=2, segment_start=2, segment_end=3, state=_state("b")),
            ]
        )


def test_validate_segments_rejects_overlapping_segments() -> None:
    with pytest.raises(ValueError, match="overlapping"):
        SegmentsLabeling(
            [
                SegmentInfo(segment_num=0, segment_start=0, segment_end=2, state=_state("a")),
                SegmentInfo(segment_num=1, segment_start=2, segment_end=4, state=_state("b")),
            ]
        )


def test_validate_segments_rejects_non_contiguous_segments() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        SegmentsLabeling(
            [
                SegmentInfo(segment_num=0, segment_start=0, segment_end=1, state=_state("a")),
                SegmentInfo(segment_num=1, segment_start=3, segment_end=4, state=_state("b")),
            ]
        )


@pytest.mark.parametrize(
    ("start", "stop", "message"),
    [
        (-1, 1, "non-negative"),
        (2, 1, "greater than or equal to start index"),
        (0, 9, "exceeds data length"),
    ],
)
def test_cut_validates_boundaries(start: int, stop: int, message: str) -> None:
    labeling = SegmentsLabeling(_segments())

    with pytest.raises(ValueError, match=message):
        labeling.cut(start, stop)


def test_cut_returns_reindexed_trimmed_segments() -> None:
    labeling = SegmentsLabeling(_segments())

    result = labeling.cut(1, 7)

    assert isinstance(result, SegmentsLabeling)
    assert [segment.segment_num for segment in result] == [0, 1, 2]
    assert [(segment.segment_start, segment.segment_end) for segment in result] == [(0, 1), (2, 4), (5, 6)]
    assert [segment.state for segment in result] == [_state("a", 1), _state("b", 2), _state("a", 1)]


def test_states_and_transitions_properties_return_unique_descriptors() -> None:
    labeling = SegmentsLabeling(_segments())

    assert labeling.states == {_state("a", 1), _state("b", 2)}
    assert labeling.transitions == {
        TransitionDescriptor(curr_state=_state("a", 1), next_state=_state("b", 2)),
        TransitionDescriptor(curr_state=_state("b", 2), next_state=_state("a", 1)),
    }


def test_query_segments_with_none_filter_returns_all_segments() -> None:
    labeling = SegmentsLabeling(_segments())

    assert labeling.query_segments(None) == list(_segments())


def test_query_bisegments_with_none_filter_returns_all_bisegments() -> None:
    labeling = SegmentsLabeling(_segments())

    assert labeling.query_bisegments(None) == [
        BisegmentInfo(
            bisegment_num=0,
            bisegment_start=0,
            bisegment_end=5,
            change_point=3,
            transition=TransitionDescriptor(curr_state=_state("a", 1), next_state=_state("b", 2)),
        ),
        BisegmentInfo(
            bisegment_num=1,
            bisegment_start=3,
            bisegment_end=8,
            change_point=6,
            transition=TransitionDescriptor(curr_state=_state("b", 2), next_state=_state("a", 1)),
        ),
    ]


def test_getitem_with_slice_returns_segments_labeling_slice() -> None:
    labeling = SegmentsLabeling(_segments())

    result = labeling[1:]

    assert isinstance(result, SegmentsLabeling)
    assert list(result) == _segments()[1:]


def test_data_len_property_returns_total_covered_length() -> None:
    labeling = SegmentsLabeling(_segments())

    assert labeling.data_len == 9


def test_merge_offsets_segment_numbers_and_data_positions() -> None:
    first = SegmentsLabeling(
        [
            SegmentInfo(segment_num=0, segment_start=0, segment_end=1, state=_state("a")),
            SegmentInfo(segment_num=1, segment_start=2, segment_end=3, state=_state("b")),
        ]
    )
    second = SegmentsLabeling(
        [
            SegmentInfo(segment_num=0, segment_start=0, segment_end=0, state=_state("c")),
            SegmentInfo(segment_num=1, segment_start=1, segment_end=2, state=_state("d")),
        ]
    )

    result = SegmentsLabeling.merge([first, second])

    assert list(result) == [
        SegmentInfo(segment_num=0, segment_start=0, segment_end=1, state=_state("a")),
        SegmentInfo(segment_num=1, segment_start=2, segment_end=3, state=_state("b")),
        SegmentInfo(segment_num=2, segment_start=4, segment_end=4, state=_state("c")),
        SegmentInfo(segment_num=3, segment_start=5, segment_end=6, state=_state("d")),
    ]


def test_from_dataframe_uses_explicit_segment_number_column() -> None:
    frame = pd.DataFrame(
        [
            {"segment": 10, "start": 0, "end": 1, "label": "a", "phase": 1},
            {"segment": 11, "start": 2, "end": 4, "label": "b", "phase": 2},
        ]
    )

    result = SegmentsLabeling.from_dataframe(frame)

    assert list(result) == [
        SegmentInfo(segment_num=10, segment_start=0, segment_end=1, state=_state("a", 1)),
        SegmentInfo(segment_num=11, segment_start=2, segment_end=4, state=_state("b", 2)),
    ]


def test_from_dataframe_uses_dataframe_index_when_segment_column_is_none() -> None:
    frame = pd.DataFrame(
        [
            {"start": 0, "end": 1, "label": "a", "phase": 1},
            {"start": 2, "end": 4, "label": "b", "phase": 2},
        ],
        index=[5, 6],
    )

    result = SegmentsLabeling.from_dataframe(frame, segment_num_col=None)

    assert list(result) == [
        SegmentInfo(segment_num=5, segment_start=0, segment_end=1, state=_state("a", 1)),
        SegmentInfo(segment_num=6, segment_start=2, segment_end=4, state=_state("b", 2)),
    ]


def test_to_dataframe_includes_segment_number_column() -> None:
    labeling = SegmentsLabeling(
        [
            SegmentInfo(segment_num=10, segment_start=0, segment_end=1, state=_state("a", 1)),
            SegmentInfo(segment_num=11, segment_start=2, segment_end=4, state=_state("b", 2)),
        ]
    )

    result = labeling.to_dataframe()

    assert result.to_dict("records") == [
        {"segment": 10, "start": 0, "end": 1, "label": "a", "phase": 1},
        {"segment": 11, "start": 2, "end": 4, "label": "b", "phase": 2},
    ]
    assert result["segment"].dtype.kind == "i"
    assert result["start"].dtype.kind == "i"
    assert result["end"].dtype.kind == "i"


def test_to_dataframe_omits_segment_number_column_when_requested() -> None:
    labeling = SegmentsLabeling(
        [
            SegmentInfo(segment_num=10, segment_start=0, segment_end=1, state=_state("a", 1)),
            SegmentInfo(segment_num=11, segment_start=2, segment_end=4, state=_state("b", 2)),
        ]
    )

    result = labeling.to_dataframe(segment_num_col=None)

    assert result.to_dict("records") == [
        {"start": 0, "end": 1, "label": "a", "phase": 1},
        {"start": 2, "end": 4, "label": "b", "phase": 2},
    ]
    assert list(result.columns) == ["start", "end", "label", "phase"]
    assert result["start"].dtype.kind == "i"
    assert result["end"].dtype.kind == "i"
