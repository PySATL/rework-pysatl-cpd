# -*- coding: ascii -*-
"""Plain (NumPy and pandas) data providers.

This subpackage provides concrete, unlabeled implementations of the
``DataProvider`` abstract interface. These providers wrap raw numeric
data stored as one-dimensional NumPy arrays, two-dimensional NumPy
arrays, or pandas DataFrames, pairing each with an annotation that
carries minimal metadata before ground-truth segment labels exist.

All providers support iteration over individual observations, length
queries, inclusive slicing via ``cut()``, and concatenation via
``merge()``. The pandas-backed provider additionally offers column
selection and derived feature creation.

.. raw:: html

    <h2>Public API</h2>

- ``NDArrayUnivariateProvider`` -- Provider for 1-D NumPy arrays
  (single-feature scalar signals). See ``np_univariate`` for details.
- ``NDArrayMultivariateProvider`` -- Provider for 2-D NumPy arrays
  (multi-feature matrix-shaped data, rows as time steps). See
  ``np_multivariate`` for details.
- ``PandasDataProvider`` -- Provider for pandas DataFrames with named
  columns. Supports ``select_columns()`` and ``create_feature_column()``.
  See ``pd_provider`` for details.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Create a univariate provider from a 1-D NumPy array:

>>> import numpy as np
>>> from pysatl_cpd.data.providers.plain import NDArrayUnivariateProvider
>>> from pysatl_cpd.data import UnlabeledTimeseriesAnnotation
>>> data = np.array([0.5, 0.7, 1.1, 0.9, 1.3], dtype=np.float64)
>>> provider = NDArrayUnivariateProvider(
...     data,
...     UnlabeledTimeseriesAnnotation(name="demo_univariate"),
... )
>>> len(provider)
5
>>> provider.raw_data.tolist()
[0.5, 0.7, 1.1, 0.9, 1.3]

Create a multivariate provider from a 2-D NumPy array:

>>> from pysatl_cpd.data.providers.plain import NDArrayMultivariateProvider
>>> matrix = np.array([[0.5, 10.0], [0.7, 10.2], [1.1, 10.4]], dtype=np.float64)
>>> mv_provider = NDArrayMultivariateProvider(
...     matrix,
...     UnlabeledTimeseriesAnnotation(name="demo_multivariate"),
... )
>>> mv_provider.raw_data.shape
(3, 2)

Create a pandas-backed provider and select columns:

>>> import pandas as pd
>>> from pysatl_cpd.data.providers.plain import PandasDataProvider
>>> df = pd.DataFrame({"value": [0.5, 0.7], "aux": [10.0, 10.2]})
>>> pd_provider = PandasDataProvider(
...     df,
...     UnlabeledTimeseriesAnnotation(name="demo_pandas"),
... )
>>> pd_provider.columns
['value', 'aux']
>>> selected = pd_provider.select_columns(["value"], rename_provider=True)
>>> selected.annotation.name
'demo_pandas[value]'

Slice a provider with inclusive boundaries and merge two providers:

>>> slice_ = provider.cut(1, 3)
>>> len(slice_)
3
>>> merged = NDArrayUnivariateProvider.merge([slice_, provider])
>>> len(merged)
8

.. raw:: html

    <h2>Notes</h2>

Notes
-----
- Change-point indices are zero-based throughout the project. The
  ``cut()`` method uses inclusive ``stop`` indices, so ``cut(0, 4)``
  returns five observations.
- All providers return copies of their underlying data via ``raw_data``
  (NumPy providers) or ``dataset`` (pandas provider) to prevent
  accidental mutation.
- ``merge()`` requires all input providers to be of the same concrete
  type. The ``|`` operator is available as a shorthand for merging two
  providers of the same type.
- ``PandasDataProvider`` iteration yields scalar values for single-column
  data and row arrays for multivariate data, matching the behavior of the
  NumPy-backed providers.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .np_multivariate import NDArrayMultivariateProvider
from .np_univariate import NDArrayUnivariateProvider
from .pd_provider import PandasDataProvider

__all__ = [
    "NDArrayMultivariateProvider",
    "NDArrayUnivariateProvider",
    "PandasDataProvider",
]
