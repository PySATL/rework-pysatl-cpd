# -*- coding: ascii -*-
"""Shared no-reset bisegment crop model and validation helpers."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class BisegmentCut:
    """Validated crop margins in sample points."""

    left_trim: int
    right_trim: int

    def __post_init__(self) -> None:
        if self.left_trim < 0 or self.right_trim < 0:
            raise ValueError("bisegment_cut margins must be non-negative")

    @property
    def is_noop(self) -> bool:
        """Return True when crop does not trim any points."""
        return self.left_trim == 0 and self.right_trim == 0

    @property
    def as_tuple(self) -> tuple[int, int]:
        """Return a tuple representation for hashing/serialization."""
        return (self.left_trim, self.right_trim)

    def validate_for_length(self, length: int) -> None:
        """Validate that the crop leaves at least one sample."""
        if self.left_trim + self.right_trim >= length:
            raise ValueError(
                "bisegment_cut trims the provider entirely: "
                f"left={self.left_trim}, right={self.right_trim}, len={length}"
            )

    @classmethod
    def parse(cls, value: BisegmentCut | tuple[int, int] | None) -> BisegmentCut:
        """Parse tuple/None input into a validated `BisegmentCut`."""
        if value is None:
            return NOOP_BISEGMENT_CUT
        if isinstance(value, cls):
            return value
        if isinstance(value, tuple):
            left_trim, right_trim = value
            return cls(left_trim=left_trim, right_trim=right_trim)
        if hasattr(value, "left_trim") and hasattr(value, "right_trim"):
            return cls(left_trim=value.left_trim, right_trim=value.right_trim)
        raise TypeError(
            f"bisegment_cut must be None, BisegmentCut, or (left_trim, right_trim); got {type(value).__name__!r}"
        )


NOOP_BISEGMENT_CUT = BisegmentCut(left_trim=0, right_trim=0)
