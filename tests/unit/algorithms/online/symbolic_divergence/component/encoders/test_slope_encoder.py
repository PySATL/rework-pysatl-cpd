# -*- coding: ascii -*-
"""
Tests for slope encoder.
"""

from __future__ import annotations

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.algorithms.online.symbolic_divergence.encoders.slope_encoder import SlopeEncoder


class TestSlopeEncoderConstruction:
    def test_rejects_negative_delta(self) -> None:
        with pytest.raises(ValueError, match="delta must be non-negative"):
            SlopeEncoder(delta=-0.1, gamma=1.0)

    def test_rejects_non_positive_gamma(self) -> None:
        with pytest.raises(ValueError, match="gamma must be positive"):
            SlopeEncoder(delta=0.0, gamma=0.0)

    def test_rejects_gamma_not_greater_than_delta(self) -> None:
        with pytest.raises(ValueError, match="strictly greater than delta"):
            SlopeEncoder(delta=1.0, gamma=1.0)


class TestSlopeEncoderProperties:
    def test_window_size_is_two(self) -> None:
        assert SlopeEncoder(delta=0.0, gamma=1.0).window_size == 2

    def test_num_symbols_is_five(self) -> None:
        assert SlopeEncoder(delta=0.0, gamma=1.0).num_symbols == 5

    def test_symbols_are_canonical_order(self) -> None:
        assert SlopeEncoder(delta=0.0, gamma=1.0).symbols == (-2, -1, 0, 1, 2)


class TestSlopeEncoderEncode:
    @pytest.mark.parametrize(
        ("window", "expected"),
        [
            ([0.0, -3.0], -2),
            ([0.0, -1.5], -1),
            ([0.0, 0.0], 0),
            ([0.0, 1.5], 1),
            ([0.0, 3.0], 2),
        ],
    )
    def test_encode_maps_difference_to_symbol(self, window: list[float], expected: int) -> None:
        encoder = SlopeEncoder(delta=0.5, gamma=2.0)
        assert encoder.encode(window) == expected

    def test_encode_uses_last_two_observations(self) -> None:
        encoder = SlopeEncoder(delta=0.0, gamma=1.0)
        assert encoder.encode([100.0, 0.0, 0.5]) == 1

    def test_encode_boundary_delta_is_flat(self) -> None:
        encoder = SlopeEncoder(delta=0.5, gamma=2.0)
        assert encoder.encode([0.0, 0.5]) == 0

    def test_encode_boundary_gamma_is_steep(self) -> None:
        encoder = SlopeEncoder(delta=0.5, gamma=2.0)
        assert encoder.encode([0.0, 2.0]) == 1

    def test_encode_rejects_short_window(self) -> None:
        encoder = SlopeEncoder(delta=0.0, gamma=1.0)
        with pytest.raises(ValueError, match="at least 2 observations"):
            encoder.encode([1.0])


class TestSlopeEncoderReset:
    def test_reset_is_noop(self) -> None:
        encoder = SlopeEncoder(delta=0.0, gamma=1.0)
        assert encoder.reset() is None
        assert encoder.encode([0.0, 2.0]) == 2
