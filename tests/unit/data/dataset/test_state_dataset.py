# -*- coding: ascii -*-
"""
Tests for state dataset.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd import (
    Dataset,
    LabeledData,
    NoChangeSeriesAnnotation,
    ProviderType,
    SegmentInfo,
    StateDataset,
    StateDescriptor,
    TimeseriesAnnotation,
)
from tests.support.providers import make_no_change_labeled, make_univariate_labeled


def test_from_dataset_rejects_non_positive_slice_length() -> None:
    dataset = Dataset([_make_segmented_series()])

    with pytest.raises(ValueError, match="slice_length must be positive, got 0"):
        StateDataset.from_dataset(dataset, slice_length=0, state=StateDescriptor(label="baseline"))


def test_from_dataset_raises_when_state_is_missing() -> None:
    dataset = Dataset([_make_segmented_series()])

    with pytest.raises(ValueError, match=r"No segments found for state label='missing'"):
        StateDataset.from_dataset(dataset, slice_length=2, state=StateDescriptor(label="missing"))


def test_from_dataset_builds_no_change_slices_and_keeps_remainder() -> None:
    state = StateDescriptor(label="baseline")
    dataset = Dataset(
        [
            _make_segmented_series(
                data=(1.0, 2.0, 3.0, 9.0, 4.0, 5.0),
                lengths=(3, 1, 2),
                labels=("baseline", "shift", "baseline"),
                name="source",
            )
        ]
    )

    result = StateDataset.from_dataset(dataset, slice_length=3, state=state, keep_remainder=True)

    assert result.state == state
    assert len(result) == 2

    first, second = result.timeseries
    assert first.annotation.provider_type is ProviderType.NO_CHANGE
    assert second.annotation.provider_type is ProviderType.NO_CHANGE
    assert first.annotation == NoChangeSeriesAnnotation(
        name="merged source[0:2]...source[4:5][state 0:2]",
        source="empty",
        state=state,
        metadata=first.annotation.metadata,
    )
    assert first.annotation.metadata["state_window_start"] == 0
    assert first.annotation.metadata["state_window_stop"] == 2
    assert second.annotation.metadata["state_window_start"] == 3
    assert second.annotation.metadata["state_window_stop"] == 4
    assert len(first) == 3
    assert len(second) == 2
    assert first.change_points == ()
    assert second.change_points == ()
    assert first.segments_labeling[0] == SegmentInfo(
        segment_num=0,
        segment_start=0,
        segment_end=2,
        state=state,
    )
    assert second.segments_labeling[0] == SegmentInfo(
        segment_num=0,
        segment_start=0,
        segment_end=1,
        state=state,
    )


def test_slice_bounds_returns_remainder_when_total_is_smaller_than_slice() -> None:
    assert StateDataset._slice_bounds(total_len=2, slice_length=5, keep_remainder=False) == []
    assert StateDataset._slice_bounds(total_len=2, slice_length=5, keep_remainder=True) == [(0, 1)]


def test_slice_bounds_appends_remainder_slice() -> None:
    assert StateDataset._slice_bounds(total_len=5, slice_length=3, keep_remainder=True) == [(0, 2), (3, 4)]


def test_slice_bounds_drops_remainder_when_not_requested() -> None:
    assert StateDataset._slice_bounds(total_len=5, slice_length=3, keep_remainder=False) == [(0, 2)]


def test_state_dataset_requires_state_for_empty_timeseries() -> None:
    with pytest.raises(ValueError, match="must contain at least one timeseries or an explicit state"):
        StateDataset([])


def test_state_dataset_accepts_explicit_state_for_empty_timeseries() -> None:
    state = StateDescriptor(label="baseline")

    dataset = StateDataset([], state=state)

    assert dataset.state == state
    assert dataset.timeseries == []


def test_state_dataset_rejects_non_no_change_provider() -> None:
    with pytest.raises(ValueError, match="requires NoChangeSeriesAnnotation providers"):
        StateDataset([_make_segmented_series()])


def test_state_dataset_rejects_non_no_change_provider_after_first_item() -> None:
    with pytest.raises(ValueError, match="requires NoChangeSeriesAnnotation providers"):
        StateDataset([make_no_change_labeled(), _make_segmented_series()])


def test_state_dataset_rejects_mismatched_provider_states() -> None:
    with pytest.raises(ValueError, match="share the same state"):
        StateDataset([make_no_change_labeled(state_label="baseline"), make_no_change_labeled(state_label="shift")])


def test_state_dataset_rejects_mismatched_explicit_state() -> None:
    provider = make_no_change_labeled(state_label="baseline")

    with pytest.raises(ValueError, match="Explicit state does not match providers state"):
        StateDataset([provider], state=StateDescriptor(label="shift"))


def test_state_dataset_preserves_explicit_state_and_builds_like() -> None:
    provider = make_no_change_labeled(state_label="baseline")
    replacement = make_no_change_labeled(data=(5.0, 6.0), name="steady-b", state_label="baseline")
    state = provider.annotation.state
    dataset = StateDataset([provider], state=state)

    rebuilt = dataset._build_like([replacement])

    assert dataset.state == state
    assert rebuilt.state == state
    assert rebuilt.timeseries == [replacement]


def _make_segmented_series(
    *,
    data: tuple[float, ...] = (1.0, 2.0, 3.0, 10.0, 11.0, 12.0),
    lengths: tuple[int, ...] = (3, 3),
    labels: tuple[str, ...] = ("baseline", "shift"),
    name: str = "series",
) -> LabeledData[float, TimeseriesAnnotation]:
    start = 0
    segments: list[SegmentInfo] = []

    for idx, (length, label) in enumerate(zip(lengths, labels, strict=True)):
        stop = start + length - 1
        segments.append(
            SegmentInfo(
                segment_num=idx,
                segment_start=start,
                segment_end=stop,
                state=StateDescriptor(label=label),
            )
        )
        start = stop + 1

    return make_univariate_labeled(data=data, name=name, segments=segments)
