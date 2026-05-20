# -*- coding: ascii -*-
"""
Annotation type definitions for time series data.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field

from pysatl_cpd.data.typedefs.provider import ProviderType
from pysatl_cpd.data.typedefs.segment import StateDescriptor, TransitionDescriptor
from pysatl_cpd.typedefs import frozendict, stable_hash

type AnnotationBuilder[AnnotationT: TimeseriesAnnotation, MergedAnnotationT: TimeseriesAnnotation] = Callable[
    [Sequence[AnnotationT]], MergedAnnotationT
]


@dataclass(frozen=True, kw_only=True, slots=True)
class TimeseriesAnnotation:
    """
    Base annotation for time series data.

    This class is the common source for all time series annotations,
    providing a name, optional source, and metadata storage.

    Attributes
    ----------
    name
        Unique identifier for this annotation.
    source
        Optional source identifier or description.
    metadata
        Additional key-value metadata storage.
    """

    name: str
    source: str | None = None
    metadata: frozendict[str, Hashable] = field(default_factory=frozendict)

    @property
    def provider_type(self) -> ProviderType:
        """
        Returns
        -------
        ProviderType
            The provider type identifier for time series annotations.
        """
        return ProviderType.TIMESERIES

    def __hash__(self) -> int:
        """Return a stable hash for the annotation identity."""
        return stable_hash((type(self).__module__, type(self).__qualname__, self.name, self.source, self.metadata))


@dataclass(frozen=True, kw_only=True, slots=True)
class UnlabeledTimeseriesAnnotation(TimeseriesAnnotation):
    """
    Annotation for unlabeled time series data.

    This annotation type represents time series without explicit
    segment or change point labels, used for unsupervised detection.

    Parameters
    ----------
    name
        Unique identifier for this annotation.
    source
        Optional source identifier or description.
    metadata
        Additional key-value metadata storage.
    """

    @property
    def provider_type(self) -> ProviderType:
        """
        Returns
        -------
        ProviderType
            The provider type identifier for unlabeled time series.
        """
        return ProviderType.UNLABELED

    def __hash__(self) -> int:
        """Return a stable hash for unlabeled annotations."""
        return stable_hash((type(self).__module__, type(self).__qualname__, self.name, self.source, self.metadata))


@dataclass(frozen=True, kw_only=True, slots=True)
class SegmentAnnotation(TimeseriesAnnotation):
    """
    Annotation with segment state information.

    This annotation type represents time series with a segment
    state descriptor, used for supervised change point detection.

    Attributes
    ----------
    state
        Descriptor for the segment state.
    """

    state: StateDescriptor

    @property
    def provider_type(self) -> ProviderType:
        """
        Returns
        -------
        ProviderType
            The provider type identifier for segment annotations.
        """
        return ProviderType.SEGMENT

    def __hash__(self) -> int:
        """Return a stable hash for segment annotations."""
        return stable_hash(
            (type(self).__module__, type(self).__qualname__, self.name, self.source, self.metadata, self.state)
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class NoChangeSeriesAnnotation(SegmentAnnotation):
    """
    Annotation for series without change points.

    This annotation type represents a time series where no change
    points are expected, used as negative examples in training.

    Parameters
    ----------
    name
        Unique identifier for this annotation.
    source
        Optional source identifier or description.
    metadata
        Additional key-value metadata storage.
    state
        Descriptor for the segment state.
    """

    @property
    def provider_type(self) -> ProviderType:
        """
        Returns
        -------
        ProviderType
            The provider type identifier for no-change series.
        """
        return ProviderType.NO_CHANGE

    def __hash__(self) -> int:
        """Return a stable hash for no-change annotations."""
        return stable_hash(
            (type(self).__module__, type(self).__qualname__, self.name, self.source, self.metadata, self.state)
        )


@dataclass(frozen=True, kw_only=True)
class BisegmentAnnotation(TimeseriesAnnotation):
    """
    Annotation with bisegment transition information.

    This annotation type represents time series with a transition
    descriptor between two segments, used for binary segmentation.

    Attributes
    ----------
    transition
        Descriptor for the segment transition.
    """

    transition: TransitionDescriptor

    @property
    def provider_type(self) -> ProviderType:
        """
        Returns
        -------
        ProviderType
            The provider type identifier for bisegment annotations.
        """
        return ProviderType.BISEGMENT

    def __hash__(self) -> int:
        """Return a stable hash for bisegment annotations."""
        return stable_hash(
            (type(self).__module__, type(self).__qualname__, self.name, self.source, self.metadata, self.transition)
        )
