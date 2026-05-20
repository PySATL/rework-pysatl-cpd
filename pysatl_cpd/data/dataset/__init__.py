# -*- coding: ascii -*-
"""Dataset types for the data layer.

Collection-level abstractions for grouping, filtering, splitting, and
merging labeled time series. This module provides the sequence interface
(IDataset), a general-purpose concrete implementation (Dataset), and a
specialized fixed-state variant (StateDataset) for no-change windows.

.. raw:: html

    <h2>Public API</h2>

- IDataset: Abstract sequence interface for labeled time series
  collections. Supports indexing, iteration, state/transition aggregation,
  train/test splitting, and merging. See idataset.py for details.
- Dataset: Concrete, backend-independent collection of labeled data with
  filtering by annotation, segments, and bisegments. See dataset.py for
  details.
- StateDataset: Dataset of fixed-state series without change points,
  typically created by slicing a larger Dataset into no-change windows.
  See state_dataset.py for details.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Create a dataset from labeled providers, inspect shared states and
transitions, split into train/test subsets, and filter by segment criteria:

>>> import numpy as np
>>> from pysatl_cpd.data.dataset import Dataset, StateDataset
>>> from pysatl_cpd.data.providers.labeled import PlainUnivariateLabeledData
>>> from pysatl_cpd.data.providers import NDArrayUnivariateProvider
>>> from pysatl_cpd.data.typedefs import (
...     SegmentInfo,
...     StateDescriptor,
...     TimeseriesAnnotation,
...     UnlabeledTimeseriesAnnotation,
... )
>>> baseline = StateDescriptor(type="baseline")
>>> shifted = StateDescriptor(type="shifted")
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

Train/test split with a reproducible seed:

>>> train, test = dataset.train_test_split(test_size=0.5, random_state=42)
>>> len(train), len(test)
(1, 1)

Filter to baseline segments only:

>>> baseline_ds = dataset.filter_by_segments(
...     lambda seg: seg.state["type"] == "baseline"
... )
>>> len(baseline_ds) > 0
True

Merge all providers into one labeled provider:

>>> merged = dataset.merge()
>>> len(merged)
8

Derive a StateDataset of no-change windows:

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
- Change-point indices are zero-based throughout. Segment boundaries
  (segment_start, segment_end) use inclusive indexing.
- Dataset filtering methods return new dataset instances; they never
  mutate the original collection.
- StateDataset requires all contained providers to have
  NoChangeSeriesAnnotation and share the same StateDescriptor.
- The train_test_split method uses the standard library random module
  for shuffling, not NumPy's RNG.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .dataset import Dataset
from .idataset import IDataset
from .state_dataset import StateDataset

__all__ = ["IDataset", "Dataset", "StateDataset"]
