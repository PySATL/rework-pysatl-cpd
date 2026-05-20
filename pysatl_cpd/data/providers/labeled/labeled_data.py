# -*- coding: ascii -*-
"""
Abstract interface for labeled data providers.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Generic, Self, TypeVar, cast

from pysatl_cpd.data.providers.data_provider import DataProvider
from pysatl_cpd.data.providers.labeled.segments_labeling import SegmentsLabeling
from pysatl_cpd.data.typedefs import (
    BisegmentAnnotation,
    BisegmentFilter,
    SegmentAnnotation,
    SegmentFilter,
    SegmentInfo,
    StateDescriptor,
    TimeseriesAnnotation,
    TransitionDescriptor,
    UnlabeledTimeseriesAnnotation,
)
from pysatl_cpd.typedefs import frozendict

DataT = TypeVar("DataT")
AnnotationT = TypeVar("AnnotationT", bound=TimeseriesAnnotation, covariant=True)


class LabeledData(Generic[DataT, AnnotationT], DataProvider[DataT, AnnotationT]):  # noqa: UP046
    """Base class for labeled sequential data.

    Parameters
    ----------
    unlabeled
        Unlabeled data provider containing the time series data.
    labeling
        Iterable of segment information for labeling.
    annotation
        Annotation associated with the labeled data.
    """

    def __init__(
        self,
        unlabeled: DataProvider[DataT, UnlabeledTimeseriesAnnotation],
        labeling: Iterable[SegmentInfo],
        annotation: AnnotationT,
    ) -> None:
        self._annotation = annotation
        self._unlabeled_data = unlabeled
        self._segments_labeling = SegmentsLabeling(list(labeling))

    @classmethod
    def from_unlabeled_data[A: TimeseriesAnnotation](
        cls,
        unlabeled: DataProvider[DataT, UnlabeledTimeseriesAnnotation],
        segment_info: Iterable[SegmentInfo],
        annotation: A,
    ) -> "LabeledData[DataT, A]":
        """Create a labeled data instance from unlabeled data.

        Alternative constructor that builds labeling from segment info.

        Parameters
        ----------
        unlabeled
            Unlabeled data provider containing the time series data.
        segment_info
            Iterable of segment information for labeling.
        annotation
            Annotation to associate with the new labeled data instance.

        Returns
        -------
        LabeledData
            LabeledData instance initialized from the provided
            unlabeled data.
        """
        return cast("LabeledData[DataT, A]", cls(unlabeled, segment_info, cast(AnnotationT, annotation)))

    @property
    def annotation(self) -> AnnotationT:
        """Return the annotation of the labeled data.

        Provides access to the associated annotation.

        Returns
        -------
        annotation
            Annotation instance for the labeled data.
        """
        return self._annotation

    @property
    def unlabeled(self) -> DataProvider[DataT, UnlabeledTimeseriesAnnotation]:
        """Return the unlabeled data provider.

        Provides access to the underlying unlabeled time series data.

        Returns
        -------
        provider
            Unlabeled data provider instance.
        """
        return self._unlabeled_data

    @property
    def segments_labeling(self) -> SegmentsLabeling:
        """Return the segments labeling information.

        Provides access to the segments labeling instance.

        Returns
        -------
        labeling
            SegmentsLabeling instance for the labeled data.
        """
        return self._segments_labeling

    @property
    def states(self) -> set[StateDescriptor]:
        """Return the set of states in the segments labeling.

        Extracts states from the underlying segments labeling.

        Returns
        -------
        states
            Set of state descriptors present in the data.
        """
        return self.segments_labeling.states

    @property
    def transitions(self) -> set[TransitionDescriptor]:
        """Return the set of transitions in the segments labeling.

        Extracts transitions from the underlying segments labeling.

        Returns
        -------
        transitions
            Set of transition descriptors present in the data.
        """
        return self.segments_labeling.transitions

    @property
    def change_points(self) -> tuple[int, ...]:
        """Return the tuple of change point indices.

        Change points are the start indices of segments after the first.

        Returns
        -------
        change_points
            Tuple of zero-based change point indices.
        """
        return tuple(info.segment_start for info in self.segments_labeling[1:])

    def cut(
        self,
        start: int,
        stop: int,
        *,
        annotation: AnnotationT | None = None,
    ) -> "LabeledData[DataT, AnnotationT]":
        """Cut a slice of the labeled data.

        Creates a new labeled data instance for the specified slice.

        Parameters
        ----------
        start
            Start index of the slice (inclusive).
        stop
            Stop index of the slice (inclusive).
        annotation
            Optional annotation for the sliced data; uses default if None.

        Returns
        -------
        labeled_data
            New labeled data instance for the sliced range.
        """
        self._unlabeled_data._validate_cut_boundaries(start, stop)

        sliced_unlabeled_data = self._unlabeled_data.cut(start, stop)
        sliced_segment_labeling = self._segments_labeling.cut(start, stop)
        sliced_annotation = annotation if annotation is not None else self.default_slice_annotation(start, stop)
        return self.from_unlabeled_data(sliced_unlabeled_data, sliced_segment_labeling, sliced_annotation)

    @classmethod
    def merge(
        cls: type[Self],
        providers: Sequence[Self],
        annotation_builder: Callable[[Sequence[AnnotationT]], AnnotationT] | None = None,
    ) -> "LabeledData[DataT, AnnotationT]":
        """Merge multiple labeled data instances into one.

        Combines unlabeled data, labeling, and annotations from providers.

        Parameters
        ----------
        providers
            Sequence of labeled data instances to merge.
        annotation_builder
            Optional callable to build merged annotation; uses
            default if None.

        Returns
        -------
        merged
            Merged labeled data instance containing all input data.
        """
        cls._validate_merge_inputs(providers)

        if annotation_builder is None:
            annotation_builder = cls.default_merge_annotation_builder()

        merged_annotation = annotation_builder([p.annotation for p in providers])
        merged_unlabeled_data = type(providers[0].unlabeled).merge([p.unlabeled for p in providers])
        merged_labeling = SegmentsLabeling.merge([p.segments_labeling for p in providers])

        return cls.from_unlabeled_data(merged_unlabeled_data, merged_labeling, merged_annotation)

    def query_segments(
        self, filter_fn: SegmentFilter | None = None
    ) -> Sequence["LabeledData[DataT, SegmentAnnotation]"]:
        """Query segments matching a filter function.

        Returns labeled data instances for each matching segment.

        Parameters
        ----------
        filter_fn
            Optional filter function to select segments; matches
            all if None.

        Returns
        -------
        segments
            Sequence of labeled data instances for matching segments.
        """
        return [
            cast(
                LabeledData[DataT, SegmentAnnotation],
                self.cut(
                    descr.segment_start,
                    descr.segment_end,
                    annotation=SegmentAnnotation(
                        name=f"{self.name}[{descr.segment_start}:{descr.segment_end}]",
                        source=self.annotation.source,
                        state=descr.state,
                        metadata=frozendict(
                            timeseries_data=self.annotation.metadata, start=descr.segment_start, end=descr.segment_end
                        ),
                    ),  # type: ignore
                ),
            )
            for descr in self._segments_labeling.query_segments(filter_fn)
        ]

    def query_bisegments(
        self, filter_fn: BisegmentFilter | None = None
    ) -> Sequence["LabeledData[DataT, BisegmentAnnotation]"]:
        """Query bisegments matching a filter function.

        Returns labeled data instances for each matching bisegment.

        Parameters
        ----------
        filter_fn
            Optional filter function to select bisegments; matches
            all if None.

        Returns
        -------
        bisegments
            Sequence of labeled data instances for matching bisegments.
        """
        return [
            cast(
                LabeledData[DataT, SegmentAnnotation],
                self.cut(
                    descr.bisegment_start,
                    descr.bisegment_end,
                    annotation=BisegmentAnnotation(
                        name=f"{self.name}[{descr.bisegment_start}:{descr.bisegment_end}]",
                        source=self.annotation.source,
                        transition=descr.transition,
                        metadata=frozendict(
                            timeseries_data=self.annotation.metadata,
                            start=descr.bisegment_start,
                            middle=descr.change_point,
                            end=descr.bisegment_end,
                        ),
                    ),  # type: ignore
                ),
            )
            for descr in self._segments_labeling.query_bisegments(filter_fn)
        ]

    def __iter__(self) -> Iterator[DataT]:
        """Iterate over the unlabeled data.

        Returns an iterator for the underlying unlabeled data points.

        Returns
        -------
        iterator
            Iterator over the unlabeled data elements.
        """
        return iter(self.unlabeled)

    def __len__(self) -> int:
        """Get the length of the unlabeled data.

        Returns the number of data points in the unlabeled provider.

        Returns
        -------
        length
            Number of data points in the unlabeled data.
        """
        return len(self.unlabeled)
