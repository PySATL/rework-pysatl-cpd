# -*- coding: ascii -*-
"""
Common type definitions for PySATL CPD data layer.

This subpackage provides the foundational type definitions used throughout the
data layer to represent time series segments, state descriptors, annotations,
provider types, and filter callables. It re-exports core numeric and immutable
mapping types from ``pysatl_cpd.typedefs`` and organizes data-specific types
into logical submodules.

.. raw:: html

    <h2>Public API</h2>

Re-exported from ``pysatl_cpd.typedefs``:

- ``Number`` -- union of NumPy scalar, int, and float for scalar numerics.
- ``NumericArray`` -- generic N-D NumPy array of numerics.
- ``frozendict[K, V_co]`` -- immutable, hashable mapping backed by
  ``MappingProxyType``.

Segment and state types (from ``segment``):

- ``StateValue`` -- type alias for valid state attribute values
  (``str | int | float | bool``).
- ``StateDescriptor`` -- immutable mapping for segment state attributes.
- ``SegmentInfo`` -- information about a time series segment including
  boundaries, segment number, and state.
- ``BisegmentInfo`` -- information about a bisegment spanning a change point
  with a transition descriptor.
- ``TransitionDescriptor`` -- describes a transition between two states.

Annotation types (from ``annotations``):

- ``TimeseriesAnnotation`` -- base annotation with name, source, and metadata.
- ``UnlabeledTimeseriesAnnotation`` -- annotation for unlabeled time series.
- ``SegmentAnnotation`` -- annotation with segment state information.
- ``NoChangeSeriesAnnotation`` -- annotation for series without change points.
- ``BisegmentAnnotation`` -- annotation with bisegment transition information.
- ``AnnotationBuilder`` -- generic callable type for merging annotations.

Provider types (from ``provider``):

- ``ProviderType`` -- StrEnum identifying data provider categories
  (segment, no_change, bisegment, timeseries, unlabeled).

Filter types (from ``filters``):

- ``SegmentFilter`` -- callable for selecting segments.
- ``BisegmentFilter`` -- callable for selecting bisegments.
- ``AnnotationFilter`` -- callable for selecting annotations.

Notes
-----
- All index values (segment boundaries, change points) are zero-based.
- Annotation and segment classes are frozen dataclasses with stable hashing
  via ``stable_hash`` from ``pysatl_cpd.typedefs``.
- ``StateDescriptor`` implements the ``Mapping`` protocol and is backed by
  ``frozendict`` for immutability and hashability.
- Filter types are type aliases for ``Callable`` and are intended for use
  with dataset and provider query methods.

Examples
--------
Create state descriptors and segment information:

>>> from pysatl_cpd.data.typedefs import StateDescriptor, SegmentInfo
>>> baseline = StateDescriptor(type="baseline", regime="stable")
>>> shifted = StateDescriptor(type="shifted", regime="changed")
>>> seg = SegmentInfo(
...     segment_num=0,
...     segment_start=0,
...     segment_end=9,
...     state=baseline,
... )
>>> seg.segment_start
0

Build a transition descriptor and bisegment info:

>>> from pysatl_cpd.data.typedefs import TransitionDescriptor, BisegmentInfo
>>> transition = TransitionDescriptor(curr_state=baseline, next_state=shifted)
>>> biseg = BisegmentInfo(
...     bisegment_num=0,
...     bisegment_start=0,
...     bisegment_end=19,
...     change_point=10,
...     transition=transition,
... )
>>> biseg.change_point
10

Create annotations with metadata:

>>> from pysatl_cpd.data.typedefs import (
...     SegmentAnnotation,
...     BisegmentAnnotation,
...     frozendict,
... )
>>> seg_ann = SegmentAnnotation(
...     name="baseline_segment",
...     state=baseline,
...     metadata=frozendict(source="synthetic"),
... )
>>> seg_ann.name
'baseline_segment'
>>> seg_ann.provider_type.value
'segment'

Use provider type enum and filter callables:

>>> from pysatl_cpd.data.typedefs import ProviderType, SegmentFilter
>>> ProviderType.SEGMENT.value
'segment'
>>> is_baseline: SegmentFilter = lambda seg: seg.state["type"] == "baseline"
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.typedefs import Number, NumericArray, frozendict

from .annotations import (
    AnnotationBuilder,
    BisegmentAnnotation,
    NoChangeSeriesAnnotation,
    SegmentAnnotation,
    TimeseriesAnnotation,
    UnlabeledTimeseriesAnnotation,
)
from .filters import (
    AnnotationFilter,
    BisegmentFilter,
    SegmentFilter,
)
from .provider import ProviderType
from .segment import (
    BisegmentInfo,
    SegmentInfo,
    StateDescriptor,
    StateValue,
    TransitionDescriptor,
)

__all__ = [
    "StateValue",
    "StateDescriptor",
    "SegmentInfo",
    "TransitionDescriptor",
    "AnnotationBuilder",
    "Number",
    "NumericArray",
    "TimeseriesAnnotation",
    "UnlabeledTimeseriesAnnotation",
    "NoChangeSeriesAnnotation",
    "SegmentAnnotation",
    "BisegmentAnnotation",
    "BisegmentInfo",
    "ProviderType",
    "SegmentFilter",
    "BisegmentFilter",
    "AnnotationFilter",
    "frozendict",
]
