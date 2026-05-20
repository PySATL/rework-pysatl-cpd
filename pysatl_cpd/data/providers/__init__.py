# -*- coding: ascii -*-
"""Public provider types for the data layer.

This package defines the core abstractions and concrete implementations for
sequential time series data in PySATL-CPD. A ``DataProvider`` is the
fundamental building block: a generic, annotated container that supports
iteration, length queries, inclusive slicing via ``cut()``, and
concatenation via ``merge()``. Concrete providers wrap raw numeric data
stored as NumPy arrays or pandas DataFrames, while labeled providers
combine observations with ordered segment descriptors to expose derived
views such as change points, states, and transitions.

Subpackages
-----------
- ``plain`` -- Concrete unlabeled providers for NumPy arrays and pandas
  DataFrames. See the ``plain`` subpackage docstring for details.
- ``labeled`` -- Abstract ``LabeledData`` interface, ``SegmentsLabeling``
  container, and concrete labeled-provider implementations. See the
  ``labeled`` subpackage docstring for details.
- ``transformers`` -- Composable feature-selection and derivation
  transformers for pandas-backed providers. See the ``transformers``
  subpackage docstring for details.

.. raw:: html

    <h2>Public API</h2>

- ``DataProvider`` -- Abstract base class for sequential data providers.
  Generic over data type and annotation type. Defines ``cut()``,
  ``merge()``, the ``|`` merge operator, and ``annotation``/``name``
  properties.
- ``NDArrayUnivariateProvider`` -- Concrete provider for 1-D NumPy arrays
  (single-feature scalar signals). Available from ``plain``.
- ``NDArrayMultivariateProvider`` -- Concrete provider for 2-D NumPy
  arrays (multi-feature matrix-shaped data). Available from ``plain``.
- ``PandasLabeledData`` -- Concrete labeled provider backed by pandas
  DataFrames with named columns. Available from ``labeled.implementations``.

Notes
-----
- Change-point indices are zero-based throughout the project. The
  ``cut()`` method uses inclusive ``stop`` indices, so ``cut(0, 4)``
  returns five observations.
- ``merge()`` requires all input providers to be of the same concrete
  type. The ``|`` operator is available as a shorthand for merging two
  providers of the same type.
- ``PandasDataProvider`` (unlabeled pandas provider) is not re-exported
  at this level; import it from ``pysatl_cpd.data.providers.plain``.

Examples
--------
Create a univariate provider from a 1-D NumPy array:

>>> import numpy as np
>>> from pysatl_cpd.data.providers import (
...     DataProvider,
...     NDArrayUnivariateProvider,
... )
>>> from pysatl_cpd.data.typedefs import UnlabeledTimeseriesAnnotation
>>> data = np.array([0.5, 0.7, 1.1, 0.9, 1.3], dtype=np.float64)
>>> provider = NDArrayUnivariateProvider(
...     data,
...     UnlabeledTimeseriesAnnotation(name="demo_univariate"),
... )
>>> len(provider)
5
>>> provider.raw_data.tolist()
[0.5, 0.7, 1.1, 0.9, 1.3]

Slice a provider with inclusive boundaries and merge two providers:

>>> slice_ = provider.cut(1, 3)
>>> len(slice_)
3
>>> merged = NDArrayUnivariateProvider.merge([slice_, provider])
>>> len(merged)
8

Use the ``|`` operator as shorthand for merging two providers:

>>> left = provider.cut(0, 1)
>>> right = provider.cut(2, 4)
>>> combined = left | right
>>> len(combined)
5

Create a labeled provider and inspect derived change points:

>>> from pysatl_cpd.data.providers import PandasLabeledData
>>> from pysatl_cpd.data.typedefs import SegmentInfo, StateDescriptor
>>> import pandas as pd
>>> from pysatl_cpd.data.providers.plain.pd_provider import PandasDataProvider
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
>>> from pysatl_cpd.data.typedefs import TimeseriesAnnotation
>>> labeled = PandasLabeledData.from_unlabeled_data(
...     unlabeled,
...     segments,
...     TimeseriesAnnotation(name="demo_labeled"),
... )
>>> list(labeled.change_points)
[2]
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .data_provider import DataProvider
from .labeled.implementations.pd_provider import PandasLabeledData
from .plain.np_multivariate import NDArrayMultivariateProvider
from .plain.np_univariate import NDArrayUnivariateProvider

__all__ = [
    "DataProvider",
    "NDArrayUnivariateProvider",
    "NDArrayMultivariateProvider",
    "PandasLabeledData",
]
