# -*- coding: ascii -*-
"""Concrete dataset implementation for labeled time series."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import logging
from collections.abc import MutableSequence, Sequence

from pysatl_cpd.data.dataset.idataset import IDataset
from pysatl_cpd.data.providers.labeled import LabeledData
from pysatl_cpd.data.typedefs import (
    AnnotationFilter,
    BisegmentAnnotation,
    BisegmentFilter,
    SegmentAnnotation,
    SegmentFilter,
    TimeseriesAnnotation,
)

logger = logging.getLogger(__name__)

__author__ = "Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


class Dataset[
    DataT,
    AnnotationT: TimeseriesAnnotation,
](IDataset[DataT, AnnotationT]):
    """Collection of labeled time series for benchmarking.

    This class provides a backend-independent collection of labeled data
    with methods for filtering by annotation, segments, and bisegments,
    as well as train/test splitting.

    Parameters
    ----------
    timeseries
        Sequence of labeled data instances.
    """

    def __init__(self, timeseries: Sequence[LabeledData[DataT, AnnotationT]]) -> None:
        super().__init__(timeseries)

    def filter_by_annotation(self, annotation_filter: AnnotationFilter | None) -> Dataset[DataT, AnnotationT]:
        """Filter dataset by annotation.

        Parameters
        ----------
        annotation_filter
            Function that takes an Annotation and returns True to include
            the labeled data.

        Returns
        -------
        Dataset
            New dataset with filtered timeseries.
        """
        annotation_filter = annotation_filter if annotation_filter is not None else lambda _: True
        filtered = [ts for ts in self._timeseries if annotation_filter(ts.annotation)]
        return type(self)(filtered)

    def filter_by_segments(self, segment_filter: SegmentFilter | None = None) -> Dataset[DataT, SegmentAnnotation]:
        """Filter dataset by segment criteria.

        Returns a new dataset where each timeseries is replaced with
        segments matching the filter, merged into new timeseries.

        Parameters
        ----------
        segment_filter
            Function that takes a SegmentInfo and returns True to include
            the segment.

        Returns
        -------
        Dataset
            New dataset with segments matching the filter.
        """
        filtered_timeseries: MutableSequence[LabeledData[DataT, SegmentAnnotation]] = []

        for ts in self._timeseries:
            segments = ts.query_segments(segment_filter)
            if segments:
                filtered_timeseries.extend(segments)

        return Dataset(filtered_timeseries)

    def filter_by_bisegments(
        self, bisegment_filter: BisegmentFilter | None = None
    ) -> Dataset[DataT, BisegmentAnnotation]:
        """
        Filter dataset by bisegment criteria.

        Returns a new dataset with all bisegments matching the filter.

        Parameters
        ----------
        bisegment_filter
            Function that takes a tuple of (current_segment, next_segment)
            and returns True to include the bisegment.

        Returns
        -------
        Dataset
            New dataset with bisegments matching the filter.
        """
        filtered_timeseries: MutableSequence[LabeledData[DataT, BisegmentAnnotation]] = []

        for ts in self._timeseries:
            bisegments = ts.query_bisegments(bisegment_filter)
            filtered_timeseries.extend(bisegments)

        return Dataset(filtered_timeseries)
