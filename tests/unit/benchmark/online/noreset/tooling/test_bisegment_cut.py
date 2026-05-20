# -*- coding: ascii -*-
"""
Tests for bisegment cut.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.benchmark.online.noreset.tooling.bisegment_cut import NOOP_BISEGMENT_CUT, BisegmentCut


class TestBisegmentCut:
    def test_negative_left_trim_raises(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            BisegmentCut(left_trim=-1, right_trim=0)

    def test_negative_right_trim_raises(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            BisegmentCut(left_trim=0, right_trim=-1)

    def test_as_tuple_returns_tuple(self) -> None:
        cut = BisegmentCut(left_trim=2, right_trim=3)
        assert cut.as_tuple == (2, 3)

    def test_validate_for_length_raises_when_trims_entirely(self) -> None:
        cut = BisegmentCut(left_trim=5, right_trim=5)
        with pytest.raises(ValueError, match="trims the provider entirely"):
            cut.validate_for_length(10)

    def test_validate_for_length_raises_when_exceeds(self) -> None:
        cut = BisegmentCut(left_trim=3, right_trim=3)
        with pytest.raises(ValueError, match="trims the provider entirely"):
            cut.validate_for_length(5)

    def test_validate_for_length_passes_when_one_sample_remains(self) -> None:
        cut = BisegmentCut(left_trim=2, right_trim=2)
        cut.validate_for_length(5)

    def test_parse_none_returns_noop(self) -> None:
        assert BisegmentCut.parse(None) is NOOP_BISEGMENT_CUT

    def test_parse_bisegment_cut_returns_self(self) -> None:
        cut = BisegmentCut(left_trim=1, right_trim=2)
        assert BisegmentCut.parse(cut) is cut

    def test_parse_tuple_creates_cut(self) -> None:
        cut = BisegmentCut.parse((3, 4))
        assert cut.left_trim == 3
        assert cut.right_trim == 4

    def test_parse_duck_typed_object(self) -> None:
        class FakeCut:
            left_trim = 5
            right_trim = 6

        cut = BisegmentCut.parse(FakeCut())
        assert cut.left_trim == 5
        assert cut.right_trim == 6

    def test_parse_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="must be None, BisegmentCut, or"):
            BisegmentCut.parse("invalid")

    def test_noop_cut_properties(self) -> None:
        assert NOOP_BISEGMENT_CUT.is_noop is True
        assert NOOP_BISEGMENT_CUT.left_trim == 0
        assert NOOP_BISEGMENT_CUT.right_trim == 0
