# -*- coding: ascii -*-
"""Generated series models."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Hashable
from dataclasses import dataclass, field

from pysatl_cpd.data.generator.segments.models import GeneratedSegment
from pysatl_cpd.data.typedefs import NumericArray, SegmentInfo, frozendict


@dataclass(frozen=True, slots=True)
class GeneratedSeries:
    """
    Generated series with data and associated metadata.

    Attributes
    ----------
    name
        Name identifier for the series, can be None.
    data
        Numeric array containing the generated time series data.
    feature_names
        Tuple of feature names corresponding to data columns.
    segments
        Tuple of segment information for each segment in the series.
    metadata
        Additional user-defined metadata as key-value pairs.
    """

    name: str | None
    data: NumericArray
    feature_names: tuple[str, ...]
    segments: tuple[SegmentInfo, ...]
    metadata: frozendict[str, Hashable] = field(default_factory=frozendict)

    @property
    def change_points(self) -> tuple[int, ...]:
        """
        Get the change points for this series.

        Returns
        -------
        change_points
            Tuple of segment end indices marking change points.
        """
        return tuple(segment.segment_end for segment in self.segments[:-1])


__all__ = ["GeneratedSegment", "GeneratedSeries"]
