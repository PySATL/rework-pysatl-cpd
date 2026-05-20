# -*- coding: ascii -*-
"""Generated segment models."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Hashable
from dataclasses import dataclass, field

from pysatl_cpd.data.typedefs import NumericArray, SegmentInfo, frozendict


@dataclass(frozen=True, slots=True)
class GeneratedSegment:
    """
    Generated segment with data and associated metadata.

    Attributes
    ----------
    name
        Name identifier for the segment.
    data
        Numeric array containing the generated time series data.
    feature_names
        Tuple of feature names corresponding to data columns.
    segment_info
        Segment metadata including start/end indices and annotations.
    metadata
        Additional user-defined metadata as key-value pairs.
    """

    name: str
    data: NumericArray
    feature_names: tuple[str, ...]
    segment_info: SegmentInfo
    metadata: frozendict[str, Hashable] = field(default_factory=frozendict)
