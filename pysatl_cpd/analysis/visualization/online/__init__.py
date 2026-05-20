# -*- coding: ascii -*-
"""
Visualization components for online change-point detection results.

This module provides visualizers and style specifications for rendering online
change-point detection results. The primary class, ``OnlineTraceVisualizer``,
produces figures showing detection function values, threshold lines, and
processing time evolution. Algorithm-specific state panels can be composed
through the ``IOnlineStateVisualizer`` interface.

Backend-agnostic style specifications (``PlotSpec``, ``LineSpec``,
``FilledLineSpec``) are re-exported here for convenience. They are defined
in the parent ``specs`` module.

.. raw:: html

    <h2>Public API</h2>

Classes
-------
OnlineTraceVisualizer
    Visualizer for online detection trace results. Renders detection function
    values, threshold lines, and processing time subplots. Supports both
    Matplotlib and Plotly backends. Accepts a composable state visualizer
    for algorithm-specific internal state panels.
IOnlineStateVisualizer
    Abstract source interface for online state visualizers. Generic over a
    type parameter bound to ``OnlineAlgorithmState``. See the ``states``
    subpackage docstring for details.
DummyStateVisualizer
    Placeholder state visualizer that performs no rendering. Useful for
    testing or when state visualization is not needed. See the ``states``
    subpackage docstring for details.
PlotSpec
    TypedDict for subplot-level options (title, axis labels, grid).
LineSpec
    TypedDict for line-style options (color, linewidth, linestyle, etc.).
FilledLineSpec
    TypedDict extending ``LineSpec`` with fill options (fill_color,
    fill_alpha).

.. raw:: html

    <h2>Subpackages</h2>

states
    Interface and implementations for algorithm state evolution visualizers.
    See the ``states`` subpackage docstring for the full API and composition
    patterns.
plotters
    High-level plotter that coordinates timeseries visualizers, trace
    visualizers, layout strategies, and annotation components. See the
    ``plotters`` subpackage docstring for details.

Examples
--------
Creating an ``OnlineTraceVisualizer`` with a dummy state visualizer::

    >>> from pysatl_cpd.analysis.visualization.online import (
    ...     DummyStateVisualizer,
    ...     OnlineTraceVisualizer,
    ... )
    >>> from pysatl_cpd.analysis.visualization.typedefs import DrawBackend
    >>> from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState
    >>> state_viz = DummyStateVisualizer[OnlineAlgorithmState](backend=DrawBackend.MATPLOTLIB)
    >>> trace_viz = OnlineTraceVisualizer(
    ...     backend=DrawBackend.MATPLOTLIB,
    ...     state_visualizer=state_viz,
    ... )
    >>> print("detection_function" in trace_viz.axes)
    True
    >>> print("processing_time" in trace_viz.axes)
    True

Configuring plot and draw options with method chaining::

    >>> trace_viz.set_detection_func_plot_opts(
    ...     title="Detection Function",
    ...     xlabel="Time Index",
    ...     ylabel="Detection Statistic",
    ... ).set_detection_func_draw_opts(
    ...     color="blue",
    ...     linewidth=1.5,
    ...     label="Detection Statistic",
    ... ).set_threshold_draw_opts(
    ...     color="red",
    ...     linestyle="dash",
    ...     linewidth=2,
    ...     alpha=0.8,
    ... )
    OnlineTraceVisualizer(...)

Using spec TypedDicts to style visual elements::

    >>> from pysatl_cpd.analysis.visualization.online import LineSpec, FilledLineSpec
    >>> line_opts: LineSpec = {"color": "green", "linewidth": 1, "label": "Threshold"}
    >>> filled_opts: FilledLineSpec = {"color": "purple", "linewidth": 1, "fill_alpha": 0.3}
    >>> print(line_opts["color"])
    green
    >>> print(filled_opts["fill_alpha"])
    0.3

Building a full visualization with ``OnlineCpdPlotter`` (from the ``plotters``
subpackage)::

    >>> from pysatl_cpd.analysis.visualization import DrawBackend, OnlineCpdPlotter
    >>> from pysatl_cpd.core.online import OnlineResetDetector
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
    ...         SegmentSpec(plan_name="a", length=100),
    ...         SegmentSpec(plan_name="b", length=80),
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
- This module requires Python 3.12+ for PEP 695 generic syntax.
- Matplotlib is required for the ``MATPLOTLIB`` backend; plotly is required
  for the ``PLOTLY`` backend.
- Change-point indices are zero-based throughout the visualization system.
- The ``OnlineTraceVisualizer`` uses a composable architecture: algorithm-
  specific state panels are added by passing an ``IOnlineStateVisualizer``
  implementation to the constructor or via ``set_state_visualizer()``.
- Style specifications (``PlotSpec``, ``LineSpec``, ``FilledLineSpec``) are
  backend-agnostic TypedDicts. They are converted to backend-specific kwargs
  by utilities in the visualization layer.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.analysis.visualization.online.online_trace_visualizer import (
    OnlineTraceVisualizer,
)
from pysatl_cpd.analysis.visualization.online.states import (
    DummyStateVisualizer,
    IOnlineStateVisualizer,
)
from pysatl_cpd.analysis.visualization.specs import FilledLineSpec, LineSpec, PlotSpec

__all__ = [
    "OnlineTraceVisualizer",
    "PlotSpec",
    "LineSpec",
    "FilledLineSpec",
    "IOnlineStateVisualizer",
    "DummyStateVisualizer",
]
