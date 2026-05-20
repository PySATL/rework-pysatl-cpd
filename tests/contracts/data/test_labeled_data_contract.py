# -*- coding: ascii -*-
"""
Tests for labeled data contract.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.data.typedefs import (
    BisegmentAnnotation,
    SegmentAnnotation,
)


class LabeledDataContract:
    """Reusable checks for LabeledData implementations."""

    @pytest.fixture
    def provider(self):
        raise NotImplementedError

    @pytest.fixture
    def other_provider(self):
        raise NotImplementedError

    def test_change_points_sorted(self, provider) -> None:
        assert tuple(sorted(provider.change_points)) == provider.change_points

    def test_change_points_non_negative(self, provider) -> None:
        assert all(change_point >= 0 for change_point in provider.change_points)

    def test_change_points_less_than_len(self, provider) -> None:
        assert all(change_point < len(provider) for change_point in provider.change_points)

    def test_segments_cover_range(self, provider) -> None:
        segments = list(provider.segments_labeling)
        assert segments[0].segment_start == 0
        assert segments[-1].segment_end == len(provider) - 1

    def test_segments_contiguous(self, provider) -> None:
        segments = list(provider.segments_labeling)
        assert all(
            left.segment_end + 1 == right.segment_start for left, right in zip(segments, segments[1:], strict=False)
        )

    def test_states_match_segment_states(self, provider) -> None:
        assert provider.states == {segment.state for segment in provider.segments_labeling}

    def test_transitions_match_adjacent_segments(self, provider) -> None:
        segments = list(provider.segments_labeling)
        expected = {(left.state, right.state) for left, right in zip(segments, segments[1:], strict=False)}
        actual = {(transition.curr_state, transition.next_state) for transition in provider.transitions}
        assert actual == expected

    def test_query_segments_without_filter_returns_all(self, provider) -> None:
        segments = provider.query_segments()
        assert len(segments) == len(list(provider.segments_labeling))
        assert all(isinstance(segment.annotation, SegmentAnnotation) for segment in segments)

    def test_query_segments_with_filter_returns_matches(self, provider) -> None:
        first_state = list(provider.segments_labeling)[0].state
        segments = provider.query_segments(lambda segment: segment.state == first_state)
        assert len(segments) == 1
        assert segments[0].annotation.state == first_state

    def test_query_bisegments_without_filter_returns_all(self, provider) -> None:
        bisegments = provider.query_bisegments()
        assert len(bisegments) == max(len(list(provider.segments_labeling)) - 1, 0)
        assert all(isinstance(segment.annotation, BisegmentAnnotation) for segment in bisegments)

    def test_query_bisegments_with_filter_returns_matches(self, provider) -> None:
        bisegments = provider.query_bisegments(lambda bisegment: bisegment.transition.next_state["label"] == "shift")
        assert len(bisegments) == 1
        assert bisegments[0].annotation.transition.next_state["label"] == "shift"

    def test_cut_within_segment_preserves_state(self, provider) -> None:
        result = provider.cut(0, 1)
        assert len(result.change_points) == 0
        assert list(result.segments_labeling)[0].state == list(provider.segments_labeling)[0].state

    def test_cut_across_boundary_shifts_change_points(self, provider) -> None:
        result = provider.cut(1, 4)
        assert result.change_points == (2,)

    def test_cut_drops_out_of_range_change_points(self, provider) -> None:
        result = provider.cut(3, 5)
        assert result.change_points == ()

    def test_unlabeled_length_matches_provider(self, provider) -> None:
        assert len(provider.unlabeled) == len(provider)

    def test_unlabeled_data_matches_provider_iteration(self, provider) -> None:
        assert len(list(provider.unlabeled)) == len(list(provider))

    def test_merge_preserves_labeling_order(self, provider, other_provider) -> None:
        merged = type(provider).merge([provider, other_provider])
        assert len(merged) == len(provider) + len(other_provider)
        assert merged.change_points[0] == provider.change_points[0]
        assert merged.change_points[-1] > len(provider) - 1

    def test_query_segments_annotation_metadata_contains_start_end(self, provider) -> None:
        segment = provider.query_segments()[0]
        assert segment.annotation.metadata["start"] == 0
        assert segment.annotation.metadata["end"] == list(provider.segments_labeling)[0].segment_end

    def test_query_bisegments_annotation_metadata_contains_middle(self, provider) -> None:
        bisegment = provider.query_bisegments()[0]
        assert bisegment.annotation.metadata["middle"] == provider.change_points[0]
