# -*- coding: ascii -*-
"""
Visualization module for change-point detection.

This module provides a backend-agnostic visualization system for rendering
time series data, online detection traces, algorithm state evolution, and
benchmark performance metrics. All visualizers and components support both
Matplotlib (static) and Plotly (interactive) backends through the
``DrawBackend`` enum.

The architecture follows two complementary patterns: the Strategy pattern
for visualizers that own complete subplots, and the Component pattern for
composable drawing elements that layer annotations onto existing axes.

.. raw:: html

    <h2>Public API</h2>

Abstract interfaces (see ``abstracts`` subpackage for details):

- ``IVisualizer``: Base interface for all visualizers managing complete
  subplots. Declares ``axes``, ``backend``, and backend-specific ``draw``
  methods.
- ``ITimeseriesVisualizer``: Interface for time series visualizers. Generic
  over ``DataProvider``.
- ``ITraceVisualizer``: Interface for detection trace visualizers. Generic
  over ``DetectionTrace``.
- ``IMetricVisualizer``: Base class for benchmark metric visualizers.
- ``IVisualComponent``: Interface for composable drawing components that
  render annotation elements onto a single subplot.

Time series visualizers (see ``timeseries`` subpackage for details):

- ``UnivariateTimeseriesVisualizer``: Renders a single time series with
  change point annotations and period fills.
- ``PlainMultivariateTimeseriesVisualizer``: Dimension-oriented visualizer
  creating one subplot per data dimension.
- ``MultivariateTimeseriesVisualizer``: Compatibility alias for
  ``PlainMultivariateTimeseriesVisualizer``.
- ``RichMultivariateTimeseriesVisualizer``: Pandas-first visualizer
  organizing data by logical plot names with optional twin y-axis.

Online detection visualizers (see ``online`` subpackage for details):

- ``OnlineTraceVisualizer``: Renders detection function values, threshold
  lines, and processing time evolution. Accepts a composable state
  visualizer for algorithm-specific panels.
- ``IOnlineStateVisualizer``: Abstract interface for algorithm state
  evolution visualizers. Generic over ``OnlineAlgorithmState``.
- ``DummyStateVisualizer``: Placeholder state visualizer for testing or
  when state visualization is not needed.

Benchmarking visualizers (see ``benchmarking`` subpackage for details):

- ``BenchmarkPlotter``: Coordinates multiple metric visualizers into
  composite benchmark figures.
- ``ThresholdBasedMetricVisualizer``: Plots metrics as functions of
  detection threshold.
- ``PrAucVisualizer``: Draws precision-recall curves with PR-AUC annotation.
- ``ARLBasedMetricVisualizer``: Plots metrics as functions of average run
  length, producing lower-envelope curves.

Coordinator and components:

- ``OnlineCpdPlotter``: High-level coordinator that bundles a timeseries
  visualizer, an online trace visualizer, and annotation components into
  a single draw call. Supports named layout strategies.
- ``VerticalLineVisualComponent``: Draws vertical lines at specified
  x-coordinates.
- ``VerticalFillComponent``: Draws filled vertical regions between pairs
  of x-coordinates.

Style specifications and types:

- ``PlotSpec``: TypedDict for subplot-level options (title, axis labels,
  grid).
- ``LineSpec``: TypedDict for line-style options (color, linewidth,
  linestyle, etc.).
- ``FilledLineSpec``: TypedDict extending ``LineSpec`` with fill options.
- ``FillSpec``: TypedDict for filled element styling.
- ``DrawBackend``: Enum with ``MATPLOTLIB`` and ``PLOTLY`` values.

Utilities:

- ``get_plotly_subplot_annotation_index``: Calculates the annotation index
  for a subplot in a Plotly figure.

.. raw:: html

    <h2>Subpackages</h2>

abstracts
    Abstract base classes defining contracts for visualizers and
    composable drawing components. See the ``abstracts`` subpackage
    docstring for architecture and backend abstraction details.

timeseries
    Univariate and multivariate time series visualizers. See the
    ``timeseries`` subpackage docstring for usage examples.

online
    Online detection trace visualizers, state visualizers, and the
    ``OnlineCpdPlotter`` coordinator. See the ``online`` subpackage
    docstring for composition patterns and layout strategies.

benchmarking
    Benchmark metric visualizers and the ``BenchmarkPlotter``
    coordinator. See the ``benchmarking`` subpackage docstring for
    registration and drawing workflow details.

components
    Reusable annotation components (vertical lines and fills). See the
    ``components`` subpackage docstring for composable usage patterns.

Examples
--------
Visualize a univariate time series with Matplotlib:

>>> import matplotlib.pyplot as plt
>>> from pysatl_cpd.analysis.visualization import (
...     DrawBackend,
...     UnivariateTimeseriesVisualizer,
... )
>>> from pysatl_cpd.data.generator import (
...     GenericSeriesGenerator,
...     ScenarioSpec,
...     SegmentPlan,
...     SegmentSpec,
...     UnivariateDistributionSpec,
...     build_plain_univariate_labeled_data,
... )
>>> from pysatl_cpd.data.typedefs import StateDescriptor, frozendict
>>> scenario = ScenarioSpec(
...     name="demo",
...     segments=(
...         SegmentSpec(plan_name="a", length=50),
...         SegmentSpec(plan_name="b", length=50),
...     ),
...     plans=frozendict(
...         a=SegmentPlan(
...             distribution=UnivariateDistributionSpec("Normal", "meanStd", mu=0.0, sigma=1.0),
...             state=StateDescriptor(type="a"),
...             name="a",
...         ),
...         b=SegmentPlan(
...             distribution=UnivariateDistributionSpec("Normal", "meanStd", mu=3.0, sigma=1.0),
...             state=StateDescriptor(type="b"),
...             name="b",
...         ),
...     ),
... )
>>> series = GenericSeriesGenerator(seed=42).generate_from_scenario(scenario, name="demo")
>>> provider = build_plain_univariate_labeled_data(series, feature_name="value", name="demo")
>>> visualizer = UnivariateTimeseriesVisualizer(backend=DrawBackend.MATPLOTLIB)
>>> visualizer.set_data_provider(provider)
>>> fig, ax = plt.subplots()
>>> visualizer.draw(figure=fig, axes={"timeseries": ax})

Add vertical line and fill components to a Plotly figure:

>>> import plotly.graph_objects as go
>>> from plotly.subplots import make_subplots
>>> from pysatl_cpd.analysis.visualization import DrawBackend
>>> from pysatl_cpd.analysis.visualization.components import (
...     VerticalFillComponent,
...     VerticalLineVisualComponent,
... )
>>> fig = make_subplots(rows=1, cols=1)
>>> _ = fig.add_trace(go.Scatter(x=list(range(10)), y=list(range(10)), name="data"), row=1, col=1)
>>> line_component = (
...     VerticalLineVisualComponent(DrawBackend.PLOTLY)
...     .set_lines([3, 7])
...     .set_style(color="red", linestyle="dash", linewidth=2, label="change points", legend=True)
... )
>>> fill_component = (
...     VerticalFillComponent(DrawBackend.PLOTLY)
...     .set_regions([(2, 4), (6, 8)])
...     .set_style(fill_color="blue", fill_alpha=0.15, label="margin", legend=True)
... )
>>> fill_component.draw(fig, (1, 1), add_legend=True)
>>> line_component.draw(fig, (1, 1), add_legend=True)

Build a full online CPD visualization with ``OnlineCpdPlotter``:

>>> from pysatl_cpd.analysis.visualization import DrawBackend, OnlineCpdPlotter
>>> from pysatl_cpd.core.online import OnlineResetDetector
>>> from pysatl_cpd.algorithms.online import ShewhartControlChart
>>> detector = OnlineResetDetector(
...     ShewhartControlChart(learning_period_size=30, window_size=10),
...     threshold=2.0,
...     skip_period=8,
... )
>>> trace = detector.detect(provider)
>>> plotter = OnlineCpdPlotter(
...     backend=DrawBackend.MATPLOTLIB,
...     data_provider=provider,
...     detection_trace=trace,
...     layout="vertical",
... )
>>> fig, ax_mapping = plotter.default_layout()
>>> fig = plotter.draw(figure=fig, axes=ax_mapping)

Notes
-----
- Requires Python 3.12+ for PEP 695 generic syntax.
- Matplotlib and plotly are required at runtime. Both are listed as
  optional dependencies in ``pyproject.toml``.
- Benchmarking visualizers additionally require pandas and numpy.
- Change-point indices are zero-based throughout the visualization
  system.
- The ``draw`` method expects an axes mapping whose keys match the
  visualizer's ``axes`` property. Matplotlib mappings use ``Axes``
  objects; Plotly mappings use ``(row, col)`` tuples.
- ``MultivariateTimeseriesVisualizer`` is a compatibility alias; prefer
  ``PlainMultivariateTimeseriesVisualizer`` in new code.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.visualization.abstracts import (
    IMetricVisualizer,
    ITimeseriesVisualizer,
    ITraceVisualizer,
    IVisualComponent,
    IVisualizer,
)
from pysatl_cpd.analysis.visualization.components import (
    VerticalFillComponent,
    VerticalLineVisualComponent,
)
from pysatl_cpd.analysis.visualization.online import (
    DummyStateVisualizer,
    FilledLineSpec,
    IOnlineStateVisualizer,
    LineSpec,
    OnlineTraceVisualizer,
    PlotSpec,
)
from pysatl_cpd.analysis.visualization.online.plotters.online_cpd_plotter import OnlineCpdPlotter
from pysatl_cpd.analysis.visualization.specs import FillSpec
from pysatl_cpd.analysis.visualization.timeseries import (
    MultivariateTimeseriesVisualizer,
    PlainMultivariateTimeseriesVisualizer,
    RichMultivariateTimeseriesVisualizer,
    UnivariateTimeseriesVisualizer,
)
from pysatl_cpd.analysis.visualization.typedefs import DrawBackend
from pysatl_cpd.analysis.visualization.utils import get_plotly_subplot_annotation_index

__all__ = [
    # Abstracts
    "IMetricVisualizer",
    "ITimeseriesVisualizer",
    "ITraceVisualizer",
    "IVisualizer",
    # Coordinator
    "OnlineCpdPlotter",
    # Backend typedefs
    "DrawBackend",
    # Time series visualizers
    "MultivariateTimeseriesVisualizer",
    "PlainMultivariateTimeseriesVisualizer",
    "RichMultivariateTimeseriesVisualizer",
    "UnivariateTimeseriesVisualizer",
    # Shared visual specs
    "PlotSpec",
    "LineSpec",
    "FillSpec",
    "FilledLineSpec",
    # Online trace visualizers
    "OnlineTraceVisualizer",
    "IOnlineStateVisualizer",
    "DummyStateVisualizer",
    # Components
    "IVisualComponent",
    # Utilities
    "get_plotly_subplot_annotation_index",
    "VerticalLineVisualComponent",
    "VerticalFillComponent",
]
