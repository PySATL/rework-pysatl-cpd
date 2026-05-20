# -*- coding: ascii -*-
"""Data-provider transformers.

Transformers provide a composable, inspectable interface for feature
preparation on data providers. Instead of ad hoc DataFrame slicing,
transformers package feature selection and derivation as reusable
objects that can be chained, inspected via their ``annotation``
property, and passed into detectors or benchmark entries.

.. raw:: html

    <h2>Public API</h2>

- ``IDataTransformer`` -- abstract base class defining the transformer
  interface (``transform`` and ``annotation``). Supports composition
  via the ``&`` operator.
- ``ComposedTransformer`` -- chains multiple transformers in sequence.
  Created explicitly or via repeated ``&`` composition.
- ``ColumnsSelectorTransformer`` -- selects a subset of feature columns
  from a ``PandasLabeledData`` provider.
- ``ColumnFeatureCreator`` -- appends a derived feature column computed
  row-wise from existing features on pandas-backed providers.

.. raw:: html

    <h2>Examples</h2>

Select feature columns from a labeled provider::

    >>> import pandas as pd
    >>> from pysatl_cpd.data import (
    ...     PandasLabeledData,
    ...     SegmentInfo,
    ...     StateDescriptor,
    ...     TimeseriesAnnotation,
    ...     UnlabeledTimeseriesAnnotation,
    ... )
    >>> from pysatl_cpd.data.providers.plain.pd_provider import PandasDataProvider
    >>> from pysatl_cpd.data.providers.transformers import ColumnsSelectorTransformer
    >>> baseline = StateDescriptor(type="baseline")
    >>> shifted = StateDescriptor(type="shifted")
    >>> df = pd.DataFrame({"value": [1.0, 1.1, 3.0, 3.1], "aux": [5.0, 5.1, 7.0, 7.1]})
    >>> unlabeled = PandasDataProvider(df, UnlabeledTimeseriesAnnotation(name="demo"))
    >>> labeled = PandasLabeledData.from_unlabeled_data(
    ...     unlabeled,
    ...     [
    ...         SegmentInfo(segment_num=0, segment_start=0, segment_end=1, state=baseline),
    ...         SegmentInfo(segment_num=1, segment_start=2, segment_end=3, state=shifted),
    ...     ],
    ...     TimeseriesAnnotation(name="demo_labeled"),
    ... )
    >>> transformer = ColumnsSelectorTransformer(columns=["value"])
    >>> result = transformer.transform(labeled)
    >>> list(result.feature_columns)
    ['value']
    >>> transformer.annotation
    'column[value]'

Derive a new feature column::

    >>> from pysatl_cpd.data.providers.transformers import ColumnFeatureCreator
    >>> creator = ColumnFeatureCreator(
    ...     name="value_sq",
    ...     mapping=lambda row: row["value"] ** 2,
    ... )
    >>> enhanced = creator.transform(labeled)
    >>> list(enhanced.feature_columns)
    ['value', 'aux', 'value_sq']

Chain transformers with the ``&`` operator::

    >>> pipeline = ColumnsSelectorTransformer(columns=["value", "aux"]) & ColumnFeatureCreator(
    ...     name="product",
    ...     mapping=lambda row: row["value"] * row["aux"],
    ... )
    >>> pipeline_result = pipeline.transform(labeled)
    >>> list(pipeline_result.feature_columns)
    ['value', 'aux', 'product']
    >>> pipeline.annotation
    'column[value;aux]->feature[product]'

Notes
-----
Transformers operate on ``DataProvider`` instances and preserve the
provider model (change points, segments, annotations). All transformers
expose an ``annotation`` property used for stable hashing and inspection.
Composition via ``&`` applies transformers left-to-right: ``a & b``
means ``a`` runs first, then ``b``.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .base import ComposedTransformer, IDataTransformer
from .columns import ColumnFeatureCreator, ColumnsSelectorTransformer

__all__ = [
    "ColumnFeatureCreator",
    "ColumnsSelectorTransformer",
    "ComposedTransformer",
    "IDataTransformer",
]
