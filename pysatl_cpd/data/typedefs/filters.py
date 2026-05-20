# -*- coding: ascii -*-
"""
Filter type definitions.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable

from pysatl_cpd.data.typedefs.annotations import TimeseriesAnnotation
from pysatl_cpd.data.typedefs.segment import BisegmentInfo, SegmentInfo

# Filter type for selecting annotations.
#
# This callable takes a TimeseriesAnnotation and returns a boolean
# indicating whether the annotation should be included in the filtered
# results.
type AnnotationFilter = Callable[[TimeseriesAnnotation], bool]
# Filter type for selecting segments.
#
# This callable takes a SegmentInfo and returns a boolean indicating
# whether the segment should be included in the filtered results.
type SegmentFilter = Callable[[SegmentInfo], bool]
# Filter type for selecting bisegments.
#
# This callable takes a BisegmentInfo and returns a boolean indicating
# whether the bisegment should be included in the filtered results.
type BisegmentFilter = Callable[[BisegmentInfo], bool]
