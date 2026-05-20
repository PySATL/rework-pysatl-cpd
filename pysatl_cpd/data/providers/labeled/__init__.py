# -*- coding: ascii -*-
"""Public labeled-provider API.

This subpackage provides the abstractions and concrete implementations for
labeled time series data. A labeled provider combines raw observations with
an ordered sequence of segment descriptors, enabling derived views such as
change points, states, transitions, per-segment slices, and per-transition
bisegment windows.

.. raw:: html

    <h2>Public API</h2>

Abstract base class:

- ``LabeledData[DataT, AnnotationT]`` -- generic base for labeled sequential
  data. Defines ``from_unlabeled_data()``, ``cut()``, ``merge()``,
  ``query_segments()``, ``query_bisegments()``, and derived properties
  ``change_points``, ``states``, and ``transitions``.

Concrete implementations (from the ``implementations`` subpackage):

- ``PlainUnivariateLabeledData`` -- NumPy-backed labeled provider for
  single-feature scalar signals.
- ``PlainMultivariateLabeledData`` -- NumPy-backed labeled provider for
  multi-feature matrix-shaped signals.
- ``PandasLabeledData`` -- Pandas-backed labeled provider with named
  columns and tabular operations (``dataset()``, ``select_columns()``,
  ``create_feature_column()``).

Re-exported type definitions (from ``pysatl_cpd.data.typedefs``):

- ``TimeseriesAnnotation`` -- base annotation for labeled time series.
- ``SegmentAnnotation`` -- annotation carrying a segment ``state``.
- ``BisegmentAnnotation`` -- annotation carrying a ``transition`` descriptor.
- ``ProviderType`` -- StrEnum identifying provider categories.
- ``StateDescriptor`` -- immutable mapping for segment state attributes.
- ``StateValue`` -- type alias for valid state attribute values.
- ``TransitionDescriptor`` -- describes a transition between two states.
- ``SegmentInfo`` -- segment boundaries, number, and state.
- ``SegmentFilter`` -- callable type for selecting segments.

Submodules and subpackages:

- ``labeled_data`` -- defines the ``LabeledData`` abstract base class.
- ``segments_labeling`` -- defines ``SegmentsLabeling``, the validated
  sequence container for ``SegmentInfo`` objects.
- ``implementations`` -- concrete labeled-data provider implementations.
  See that subpackage's docstring for detailed usage examples.

Notes
-----
- All index values (segment boundaries, change points) are zero-based.
- Segment labeling must be contiguous and non-overlapping; violations
  raise ``ValueError`` during construction.
- The ``from_unlabeled_data()`` class method is the preferred constructor.
  It validates that the unlabeled provider matches the expected backend
  type and raises ``TypeError`` otherwise.
- Type definitions are re-exported from ``pysatl_cpd.data.typedefs``.
  See that subpackage's docstring for full details on annotations,
  descriptors, and filter types.

Examples
--------
Build a univariate labeled provider from raw data and segment info:

>>> import numpy as np
>>> from pysatl_cpd.data.providers.labeled import (
...     LabeledData,
...     PlainUnivariateLabeledData,
...     SegmentInfo,
...     StateDescriptor,
...     TimeseriesAnnotation,
... )
>>> from pysatl_cpd.data.providers.plain.np_univariate import (
...     NDArrayUnivariateProvider,
... )
>>> from pysatl_cpd.data.typedefs import UnlabeledTimeseriesAnnotation
>>> baseline = StateDescriptor(type="baseline")
>>> shifted = StateDescriptor(type="shifted")
>>> data = np.array([0.1, 0.2, 0.0, 3.0, 3.1, 2.9], dtype=np.float64)
>>> unlabeled = NDArrayUnivariateProvider(
...     data,
...     UnlabeledTimeseriesAnnotation(name="demo"),
... )
>>> segments = [
...     SegmentInfo(segment_num=0, segment_start=0, segment_end=2, state=baseline),
...     SegmentInfo(segment_num=1, segment_start=3, segment_end=5, state=shifted),
... ]
>>> labeled = PlainUnivariateLabeledData.from_unlabeled_data(
...     unlabeled,
...     segments,
...     TimeseriesAnnotation(name="demo_labeled"),
... )
>>> list(labeled.change_points)
[3]
>>> [dict(s) for s in labeled.states]
[{'type': 'baseline'}, {'type': 'shifted'}]

Query segments and bisegments:

>>> baseline_segs = labeled.query_segments(
...     lambda seg: seg.state["type"] == "baseline"
... )
>>> len(baseline_segs)
1
>>> bisegments = labeled.query_bisegments()
>>> len(bisegments)
1
>>> bisegments[0].annotation.transition.curr_state == baseline
True

Cut a slice and merge providers:

>>> sliced = labeled.cut(1, 4)
>>> len(sliced)
4
>>> merged = type(labeled).merge([sliced, labeled])
>>> len(merged)
10
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.data.typedefs import (
    BisegmentAnnotation,
    ProviderType,
    SegmentAnnotation,
    SegmentFilter,
    SegmentInfo,
    StateDescriptor,
    StateValue,
    TimeseriesAnnotation,
    TransitionDescriptor,
)

from .implementations import PandasLabeledData, PlainMultivariateLabeledData, PlainUnivariateLabeledData
from .labeled_data import LabeledData

__all__ = [
    "LabeledData",
    "TimeseriesAnnotation",
    "SegmentAnnotation",
    "BisegmentAnnotation",
    "ProviderType",
    "StateDescriptor",
    "StateValue",
    "TransitionDescriptor",
    "SegmentInfo",
    "SegmentFilter",
    "PlainUnivariateLabeledData",
    "PlainMultivariateLabeledData",
    "PandasLabeledData",
]
