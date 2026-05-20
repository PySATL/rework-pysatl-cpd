# -*- coding: ascii -*-
"""Provider builders for generated data.

Converts ``GeneratedSeries`` objects produced by the generator API into
labeled-provider types used by the data layer and downstream detectors.

.. raw:: html

    <h2>Public API</h2>

- ``build_pandas_labeled_data`` -- Build a multivariate ``PandasLabeledData``
  from a ``GeneratedSeries``.
- ``build_pandas_univariate_labeled_data`` -- Build a univariate
  ``PandasLabeledData`` by selecting one named feature from a
  ``GeneratedSeries``.
- ``build_plain_multivariate_labeled_data`` -- Build a multivariate
  ``PlainMultivariateLabeledData`` (backed by NumPy arrays) from a
  ``GeneratedSeries``.
- ``build_plain_univariate_labeled_data`` -- Build a univariate
  ``PlainUnivariateLabeledData`` (backed by NumPy arrays) by selecting one
  named feature from a ``GeneratedSeries``.

.. raw:: html

    <h2>Submodules</h2>

- ``np_provider`` -- Plain (NumPy-backed) provider builders.
- ``pd_provider`` -- Pandas-backed provider builders.

.. raw:: html

    <h2>Examples</h2>

Build a pandas multivariate labeled provider from a generated series::

    >>> from pysatl_cpd.data.generator import (
    ...     GenericSeriesGenerator,
    ...     NormalSpec,
    ...     ScenarioSpec,
    ...     SegmentPlan,
    ...     SegmentSpec,
    ...     build_pandas_labeled_data,
    ... )
    >>> from pysatl_cpd.data.typedefs import StateDescriptor, frozendict
    >>> scenario = ScenarioSpec(
    ...     name="example",
    ...     segments=(
    ...         SegmentSpec(plan_name="a", length=50),
    ...         SegmentSpec(plan_name="b", length=30),
    ...     ),
    ...     plans=frozendict(
    ...         a=SegmentPlan(
    ...             distribution=NormalSpec(mean=0.0, std=1.0),
    ...             state=StateDescriptor(type="baseline"),
    ...         ),
    ...         b=SegmentPlan(
    ...             distribution=NormalSpec(mean=3.0, std=1.0),
    ...             state=StateDescriptor(type="shifted"),
    ...         ),
    ...     ),
    ... )
    >>> series = GenericSeriesGenerator(seed=0).generate_from_scenario(scenario)
    >>> provider = build_pandas_labeled_data(series, name="example")
    >>> list(provider.feature_columns)
    ['value']
    >>> provider.change_points
    (50,)

Build a plain univariate labeled provider by selecting a single feature::

    >>> from pysatl_cpd.data.generator import (
    ...     GenericSeriesGenerator,
    ...     IndependentColumnsSpec,
    ...     ScenarioSpec,
    ...     SegmentPlan,
    ...     SegmentSpec,
    ...     build_plain_univariate_labeled_data,
    ... )
    >>> from pysatl_cpd.data.typedefs import StateDescriptor, frozendict
    >>> scenario = ScenarioSpec(
    ...     name="multi_example",
    ...     segments=(
    ...         SegmentSpec(plan_name="a", length=40),
    ...         SegmentSpec(plan_name="b", length=20),
    ...     ),
    ...     plans=frozendict(
    ...         a=SegmentPlan(
    ...             distribution=IndependentColumnsSpec(
    ...                 columns=frozendict(
    ...                     x=NormalSpec(mean=0.0, std=1.0),
    ...                     y=NormalSpec(mean=10.0, std=2.0),
    ...                 ),
    ...             ),
    ...             state=StateDescriptor(type="baseline"),
    ...         ),
    ...         b=SegmentPlan(
    ...             distribution=IndependentColumnsSpec(
    ...                 columns=frozendict(
    ...                     x=NormalSpec(mean=5.0, std=1.0),
    ...                     y=NormalSpec(mean=15.0, std=2.0),
    ...                 ),
    ...             ),
    ...             state=StateDescriptor(type="shifted"),
    ...         ),
    ...     ),
    ... )
    >>> series = GenericSeriesGenerator(seed=0).generate_from_scenario(scenario)
    >>> provider = build_plain_univariate_labeled_data(
    ...     series, feature_name="y", name="multi_example",
    ... )
    >>> provider.raw_data.shape
    (60,)
    >>> provider.change_points
    (40,)

Notes
-----
All builders accept an optional ``annotation`` argument. When omitted, a
``TimeseriesAnnotation`` is constructed automatically from the series metadata
and the supplied ``name``.

The univariate builders raise ``ValueError`` if the requested ``feature_name``
is not present in the generated series.

Change-point indices returned by the resulting labeled providers are zero-based.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .np_provider import build_plain_multivariate_labeled_data, build_plain_univariate_labeled_data
from .pd_provider import build_pandas_labeled_data, build_pandas_univariate_labeled_data

__all__ = [
    "build_pandas_labeled_data",
    "build_pandas_univariate_labeled_data",
    "build_plain_multivariate_labeled_data",
    "build_plain_univariate_labeled_data",
]
