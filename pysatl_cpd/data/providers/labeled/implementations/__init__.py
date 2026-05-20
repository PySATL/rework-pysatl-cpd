# -*- coding: ascii -*-
"""Concrete labeled-data provider implementations.

This module provides the three concrete implementations of the
``LabeledData`` abstract base class, each backed by a different storage
format. These classes combine raw time series data with ordered segment
labeling to produce fully labeled providers that expose change points,
states, transitions, and segment-aware query operations.

.. raw:: html

    <h2>Public API</h2>

- ``PlainUnivariateLabeledData`` -- NumPy-backed labeled provider for
  single-feature scalar signals. Wraps ``NDArrayUnivariateProvider``.
- ``PlainMultivariateLabeledData`` -- NumPy-backed labeled provider for
  multi-feature matrix-shaped signals. Wraps ``NDArrayMultivariateProvider``.
- ``PandasLabeledData`` -- Pandas-backed labeled provider with named
  columns and table-oriented operations. Wraps ``PandasDataProvider``.

All three classes share the inherited ``LabeledData`` interface, including
``from_unlabeled_data()``, ``cut()``, ``merge()``, ``query_segments()``,
``query_bisegments()``, and derived properties ``change_points``,
``states``, and ``transitions``. ``PandasLabeledData`` adds
``dataset()``, ``feature_columns``, ``select_columns()``, and
``create_feature_column()`` for tabular workflows.

Notes
-----
- All index values (segment boundaries, change points) are zero-based.
- The ``from_unlabeled_data()`` class method is the preferred constructor.
  It validates that the unlabeled provider matches the expected backend
  type and raises ``TypeError`` otherwise.
- ``PandasLabeledData`` accepts optional keyword arguments
  ``segment_column``, ``segment_start_column``, and ``segment_end_column``
  to customize the names of segment columns in the output DataFrame.
- Segment labeling must be contiguous and non-overlapping; violations
  raise ``ValueError`` during construction.

Examples
--------
Create a univariate labeled provider from raw data and segment info:

>>> import numpy as np
>>> from pysatl_cpd.data.providers.labeled.implementations import (
...     PlainUnivariateLabeledData,
... )
>>> from pysatl_cpd.data.providers.plain.np_univariate import (
...     NDArrayUnivariateProvider,
... )
>>> from pysatl_cpd.data.typedefs import (
...     SegmentInfo,
...     StateDescriptor,
...     TimeseriesAnnotation,
...     UnlabeledTimeseriesAnnotation,
... )
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

Create a multivariate labeled provider:

>>> from pysatl_cpd.data.providers.labeled.implementations import (
...     PlainMultivariateLabeledData,
... )
>>> from pysatl_cpd.data.providers.plain.np_multivariate import (
...     NDArrayMultivariateProvider,
... )
>>> mv_data = np.array(
...     [[0.0, 10.0], [0.1, 10.2], [3.0, 20.0], [2.9, 19.8]],
...     dtype=np.float64,
... )
>>> mv_unlabeled = NDArrayMultivariateProvider(
...     mv_data,
...     UnlabeledTimeseriesAnnotation(name="mv_demo"),
... )
>>> mv_segments = [
...     SegmentInfo(segment_num=0, segment_start=0, segment_end=1, state=baseline),
...     SegmentInfo(segment_num=1, segment_start=2, segment_end=3, state=shifted),
... ]
>>> mv_labeled = PlainMultivariateLabeledData.from_unlabeled_data(
...     mv_unlabeled,
...     mv_segments,
...     TimeseriesAnnotation(name="mv_labeled"),
... )
>>> mv_labeled.raw_data.shape
(4, 2)

Create a pandas-backed labeled provider and use tabular operations:

>>> import pandas as pd
>>> from pysatl_cpd.data.providers.labeled.implementations import PandasLabeledData
>>> from pysatl_cpd.data.providers.plain.pd_provider import PandasDataProvider
>>> df = pd.DataFrame({"value": [0.1, 0.2, 3.0, 3.1], "aux": [1.0, 1.1, 2.0, 2.1]})
>>> pd_unlabeled = PandasDataProvider(
...     df,
...     UnlabeledTimeseriesAnnotation(name="pd_demo"),
... )
>>> pd_segments = [
...     SegmentInfo(segment_num=0, segment_start=0, segment_end=1, state=baseline),
...     SegmentInfo(segment_num=1, segment_start=2, segment_end=3, state=shifted),
... ]
>>> pd_labeled = PandasLabeledData.from_unlabeled_data(
...     pd_unlabeled,
...     pd_segments,
...     TimeseriesAnnotation(name="pd_labeled"),
... )
>>> list(pd_labeled.feature_columns)
['value', 'aux']
>>> selected = pd_labeled.select_columns(feature_columns=["value"])
>>> list(selected.feature_columns)
['value']
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .np_multivariate import PlainMultivariateLabeledData
from .np_univariate import PlainUnivariateLabeledData
from .pd_provider import PandasLabeledData

__all__ = [
    "PlainMultivariateLabeledData",
    "PlainUnivariateLabeledData",
    "PandasLabeledData",
]
