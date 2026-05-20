# -*- coding: ascii -*-
"""Public data-layer API for PySATL-CPD.

This module provides the foundational data abstractions used throughout the
PySATL change-point detection project. It turns raw arrays, tables, and
labels into a stable vocabulary that detectors, benchmarks, and generators
all share. Change-point detectors do not need to care whether a signal
started as a NumPy array or a pandas DataFrame; benchmark code does not
need to care how states or transitions were encoded. The data layer is
where those concerns are resolved once, in one consistent shape.

The module exposes four layers of abstraction:

1. **Providers** -- Sequential containers for unlabeled and labeled time
   series data. Unlabeled providers (NumPy-backed or pandas-backed) pair
   raw observations with minimal metadata. Labeled providers add ordered
   segment descriptors and expose derived views such as change points,
   states, transitions, per-segment slices, and per-transition bisegment
   windows.

2. **Datasets** -- Collection-level abstractions for grouping, filtering,
   splitting, and merging labeled time series. Datasets behave like
   sequences of labeled providers while adding collection-level semantics
   such as state/transition aggregation, train/test splitting, and
   dataset-wide filtering.

3. **Loaders** -- File-backed data loading utilities that turn on-disk
   segmented CSV files into in-memory Dataset objects.

4. **Type definitions** -- Foundational types for segments, states,
   annotations, provider categories, and filter callables.

.. raw:: html

    <h2>Public API</h2>

Providers (from ``providers`` and ``providers.labeled``):

- ``DataProvider`` -- Abstract base class for sequential data providers.
  Generic over data type and annotation type. Defines ``cut()``,
  ``merge()``, the ``|`` merge operator, and ``annotation``/``name``
  properties.
- ``NDArrayUnivariateProvider`` -- Concrete provider for 1-D NumPy arrays
  (single-feature scalar signals).
- ``NDArrayMultivariateProvider`` -- Concrete provider for 2-D NumPy
  arrays (multi-feature matrix-shaped data).
- ``PandasLabeledData`` -- Concrete labeled provider backed by pandas
  DataFrames with named columns.
- ``LabeledData`` -- Generic abstract base class for labeled sequential
  data. Defines ``from_unlabeled_data()``, ``cut()``, ``merge()``,
  ``query_segments()``, ``query_bisegments()``, and derived properties
  ``change_points``, ``states``, and ``transitions``.
- ``PlainUnivariateLabeledData`` -- NumPy-backed labeled provider for
  single-feature scalar signals.
- ``PlainMultivariateLabeledData`` -- NumPy-backed labeled provider for
  multi-feature matrix-shaped signals.

Datasets (from ``dataset``):

- ``IDataset`` -- Abstract sequence interface for labeled time series
  collections. Supports indexing, iteration, state/transition aggregation,
  train/test splitting, and merging.
- ``Dataset`` -- Concrete, backend-independent collection of labeled data
  with filtering by annotation, segments, and bisegments.
- ``StateDataset`` -- Dataset of fixed-state series without change points,
  typically created by slicing a larger Dataset into no-change windows.

Loaders (from ``loaders``):

- ``FolderCsvColumns`` -- Immutable configuration specifying which CSV
  columns contain feature data, state labels, and segment numbers.
- ``load_folder_csv_dataset`` -- Load a ``Dataset`` from a root directory
  of segmented CSV folders, returning a collection of
  ``PandasLabeledData`` providers.

Type definitions (from ``typedefs``):

- ``StateValue`` -- Type alias for valid state attribute values.
- ``StateDescriptor`` -- Immutable mapping for segment state attributes.
- ``SegmentInfo`` -- Information about a time series segment including
  boundaries, segment number, and state.
- ``BisegmentInfo`` -- Information about a bisegment spanning a change
  point with a transition descriptor.
- ``TransitionDescriptor`` -- Describes a transition between two states.
- ``TimeseriesAnnotation`` -- Base annotation with name, source, and
  metadata.
- ``UnlabeledTimeseriesAnnotation`` -- Annotation for unlabeled time
  series.
- ``SegmentAnnotation`` -- Annotation with segment state information.
- ``NoChangeSeriesAnnotation`` -- Annotation for series without change
  points.
- ``BisegmentAnnotation`` -- Annotation with bisegment transition
  information.
- ``AnnotationBuilder`` -- Generic callable type for merging annotations.
- ``ProviderType`` -- StrEnum identifying data provider categories.
- ``SegmentFilter`` -- Callable type for selecting segments.
- ``BisegmentFilter`` -- Callable type for selecting bisegments.

.. raw:: html

    <h2>Subpackages</h2>

- ``providers`` -- Core provider abstractions and concrete implementations
  for unlabeled and labeled data. See its docstring for detailed usage.
- ``providers.plain`` -- Concrete unlabeled providers for NumPy arrays and
  pandas DataFrames. See its docstring for details.
- ``providers.labeled`` -- Abstract ``LabeledData`` interface and concrete
  labeled-provider implementations. See its docstring for details.
- ``dataset`` -- Dataset types (``IDataset``, ``Dataset``, ``StateDataset``).
  See its docstring for detailed usage examples.
- ``loaders`` -- File-backed CSV loading utilities. See its docstring for
  details.
- ``typedefs`` -- Foundational type definitions for segments, states,
  annotations, and filters. See its docstring for details.
- ``generator`` -- Synthetic data-generation API for creating scenarios,
  series, and datasets with known regime structure. See its docstring for
  detailed usage.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Create an unlabeled univariate provider and inspect its data:

>>> import numpy as np
>>> from pysatl_cpd.data import (
...     NDArrayUnivariateProvider,
...     UnlabeledTimeseriesAnnotation,
... )
>>> data = np.array([0.5, 0.7, 1.1, 0.9, 1.3], dtype=np.float64)
>>> provider = NDArrayUnivariateProvider(
...     data,
...     UnlabeledTimeseriesAnnotation(name="demo_univariate"),
... )
>>> len(provider)
5
>>> provider.raw_data.tolist()
[0.5, 0.7, 1.1, 0.9, 1.3]

Build a labeled provider from raw data and segment descriptors:

>>> from pysatl_cpd.data import (
...     PandasLabeledData,
...     SegmentInfo,
...     StateDescriptor,
...     TimeseriesAnnotation,
... )
>>> from pysatl_cpd.data.providers.plain.pd_provider import PandasDataProvider
>>> import pandas as pd
>>> baseline = StateDescriptor(type="baseline")
>>> shifted = StateDescriptor(type="shifted")
>>> df = pd.DataFrame({"value": [0.1, 0.2, 3.0, 3.1]})
>>> unlabeled = PandasDataProvider(
...     df,
...     UnlabeledTimeseriesAnnotation(name="demo"),
... )
>>> segments = [
...     SegmentInfo(segment_num=0, segment_start=0, segment_end=1, state=baseline),
...     SegmentInfo(segment_num=1, segment_start=2, segment_end=3, state=shifted),
... ]
>>> labeled = PandasLabeledData.from_unlabeled_data(
...     unlabeled,
...     segments,
...     TimeseriesAnnotation(name="demo_labeled"),
... )
>>> list(labeled.change_points)
[2]

Query segments and bisegments from a labeled provider:

>>> baseline_segs = labeled.query_segments(
...     lambda seg: seg.state["type"] == "baseline"
... )
>>> len(baseline_segs)
1
>>> bisegments = labeled.query_bisegments()
>>> len(bisegments)
1

Slice a labeled provider with inclusive boundaries and merge:

>>> sliced = labeled.cut(1, 3)
>>> len(sliced)
3
>>> merged = type(labeled).merge([sliced, labeled])
>>> len(merged)
7

Create a dataset from multiple labeled providers:

>>> from pysatl_cpd.data import (
...     Dataset,
...     PlainUnivariateLabeledData,
... )
>>> series_a = PlainUnivariateLabeledData.from_unlabeled_data(
...     NDArrayUnivariateProvider(
...         np.array([0.1, 0.3, -0.2, 0.4], dtype=np.float64),
...         UnlabeledTimeseriesAnnotation(name="series_a"),
...     ),
...     [
...         SegmentInfo(segment_num=0, segment_start=0, segment_end=1, state=baseline),
...         SegmentInfo(segment_num=1, segment_start=2, segment_end=3, state=shifted),
...     ],
...     TimeseriesAnnotation(name="series_a"),
... )
>>> series_b = PlainUnivariateLabeledData.from_unlabeled_data(
...     NDArrayUnivariateProvider(
...         np.array([-0.1, 2.9, 3.1, 2.8], dtype=np.float64),
...         UnlabeledTimeseriesAnnotation(name="series_b"),
...     ),
...     [
...         SegmentInfo(segment_num=0, segment_start=0, segment_end=0, state=baseline),
...         SegmentInfo(segment_num=1, segment_start=1, segment_end=3, state=shifted),
...     ],
...     TimeseriesAnnotation(name="series_b"),
... )
>>> dataset = Dataset([series_a, series_b])
>>> len(dataset)
2
>>> sorted(dataset.states, key=lambda s: s["type"])
[type='baseline', type='shifted']

Split a dataset into train/test subsets and filter by segment state:

>>> train, test = dataset.train_test_split(test_size=0.5, random_state=42)
>>> len(train), len(test)
(1, 1)
>>> baseline_ds = dataset.filter_by_segments(
...     lambda seg: seg.state["type"] == "baseline"
... )
>>> len(baseline_ds) > 0
True

Derive a StateDataset of no-change windows:

>>> from pysatl_cpd.data import StateDataset
>>> state_ds = StateDataset.from_dataset(
...     dataset,
...     slice_length=2,
...     state=baseline,
...     keep_remainder=True,
... )
>>> len(state_ds) > 0
True
>>> state_ds.state
type='baseline'

.. raw:: html

    <h2>Notes</h2>

Notes
-----
- Change-point indices are zero-based throughout the project. The
  ``cut()`` method uses inclusive ``stop`` indices, so ``cut(0, 4)``
  returns five observations.
- ``merge()`` requires all input providers to be of the same concrete
  type. The ``|`` operator is available as a shorthand for merging two
  providers of the same type.
- Dataset filtering methods return new dataset instances; they never
  mutate the original collection.
- ``StateDataset`` requires all contained providers to have
  ``NoChangeSeriesAnnotation`` and share the same ``StateDescriptor``.
- All providers return copies of their underlying data to prevent
  accidental mutation.
- For programmatic data creation without file I/O, use the generator
  API in ``pysatl_cpd.data.generator``.
- The ``train_test_split`` method uses the standard library ``random``
  module for shuffling, not NumPy's RNG.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .dataset import Dataset, IDataset, StateDataset
from .loaders import FolderCsvColumns, load_folder_csv_dataset
from .providers import DataProvider, NDArrayMultivariateProvider, NDArrayUnivariateProvider, PandasLabeledData
from .providers.labeled import LabeledData, PlainMultivariateLabeledData, PlainUnivariateLabeledData
from .typedefs import (
    AnnotationBuilder,
    BisegmentAnnotation,
    BisegmentFilter,
    BisegmentInfo,
    NoChangeSeriesAnnotation,
    ProviderType,
    SegmentAnnotation,
    SegmentFilter,
    SegmentInfo,
    StateDescriptor,
    StateValue,
    TimeseriesAnnotation,
    TransitionDescriptor,
    UnlabeledTimeseriesAnnotation,
)

__all__ = [
    "AnnotationBuilder",
    "BisegmentAnnotation",
    "BisegmentFilter",
    "BisegmentInfo",
    "DataProvider",
    "Dataset",
    "FolderCsvColumns",
    "IDataset",
    "LabeledData",
    "load_folder_csv_dataset",
    "NDArrayMultivariateProvider",
    "NDArrayUnivariateProvider",
    "NoChangeSeriesAnnotation",
    "PandasLabeledData",
    "PlainMultivariateLabeledData",
    "PlainUnivariateLabeledData",
    "ProviderType",
    "SegmentAnnotation",
    "SegmentFilter",
    "SegmentInfo",
    "StateDataset",
    "StateDescriptor",
    "StateValue",
    "TimeseriesAnnotation",
    "TransitionDescriptor",
    "UnlabeledTimeseriesAnnotation",
]
