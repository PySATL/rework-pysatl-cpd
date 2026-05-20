# -*- coding: ascii -*-
"""
Benchmark visualization for change-point detection algorithms.

This package provides tools for rendering benchmark comparison results as
publication-ready plots. It coordinates multiple metric visualizers through
a central ``BenchmarkPlotter``, which manages registration, data propagation,
and composite drawing across Matplotlib and Plotly backends. Concrete metric
visualizer implementations consume precomputed benchmark tables (pandas
DataFrames) and draw metric curves such as F1 vs threshold, precision-recall
trajectories, ARL envelopes, and PR-AUC annotated curves.

.. raw:: html

    <h2>Public API</h2>

- ``BenchmarkPlotter``: Coordinates benchmark metric visualizers. Manages
  registration, data propagation, and composite drawing across matplotlib
  and plotly backends.
- ``IMetricVisualizer``: Abstract base class defining the contract for metric
  visualizers. Manages benchmark tables and per-entry line styling.
- ``MetricVisualizerName``: Type alias (``str``) for metric visualizer names
  used as keys in registration and axes mapping.
- ``MetricPlotName``: Type alias (``str``) for metric plot names.
- ``PrAucVisualizer``: Draws precision-recall curves with PR-AUC score
  annotation.
- ``ThresholdBasedMetricVisualizer``: Plots metrics as functions of detection
  threshold, with optional monotonic-precision mode.
- ``ARLBasedMetricVisualizer``: Plots metrics as functions of average run
  length (ARL), producing lower-envelope curves.

.. raw:: html

    <h2>Subpackages</h2>

- ``metrics``: Concrete ``IMetricVisualizer`` implementations
  (``ThresholdBasedMetricVisualizer``, ``PrAucVisualizer``,
  ``ARLBasedMetricVisualizer``). See that subpackage's docstring for details
  on each visualizer's data requirements and rendering behavior.
- ``plotters``: The ``BenchmarkPlotter`` coordinator and associated type
  aliases. See that subpackage's docstring for composition and drawing
  workflow details.

.. raw:: html

    <h2>Examples</h2>

Register metric visualizers and benchmark tables, then draw into a
matplotlib figure::

    >>> import matplotlib.pyplot as plt
    >>> import pandas as pd
    >>> from pysatl_cpd.analysis.visualization.benchmarking import (
    ...     BenchmarkPlotter,
    ...     ThresholdBasedMetricVisualizer,
    ... )
    >>> benchmark_tables = {
    ...     "shewhart": pd.DataFrame({
    ...         "threshold": [1.0, 2.0, 3.0],
    ...         "f1": [0.8, 0.9, 0.95],
    ...     }),
    ... }
    >>> plotter = (
    ...     BenchmarkPlotter()
    ...     .set_metrics(
    ...         {
    ...             "f1": ThresholdBasedMetricVisualizer(
    ...                 y_metrics=["f1"], title="F1", ylabel="F1"
    ...             ),
    ...         }
    ...     )
    ...     .set_benchmark_tables(benchmark_tables)
    ... )
    >>> fig, axes = plt.subplots(1, 1)
    >>> fig = plotter.draw(figure=fig, axes={"f1": axes})

The same plotter works with plotly by passing a ``plotly.graph_objects.Figure``
and a subplot-position mapping::

    >>> import plotly.graph_objects as go
    >>> from plotly.subplots import make_subplots
    >>> from pysatl_cpd.analysis.visualization import DrawBackend
    >>> plotter_go = (
    ...     BenchmarkPlotter()
    ...     .set_metrics(
    ...         {
    ...             "f1": ThresholdBasedMetricVisualizer(
    ...                 backend=DrawBackend.PLOTLY,
    ...                 y_metrics=["f1"], title="F1", ylabel="F1",
    ...             ),
    ...         }
    ...     )
    ...     .set_benchmark_tables(benchmark_tables)
    ... )
    >>> fig = make_subplots(rows=1, cols=1)
    >>> fig = plotter_go.draw(figure=fig, axes={"f1": (1, 1)})

Line styles can be customized per entry or per entry-metric pair before
drawing::

    >>> plotter.set_entry_draw_opts(
    ...     entry="shewhart", color="blue", linewidth=2
    ... )

Notes
-----
- Requires ``pandas``, ``matplotlib``, and ``plotly`` at runtime.
- ``IMetricVisualizer`` is imported from the parent ``abstracts`` subpackage
  (``pysatl_cpd.analysis.visualization.abstracts``).
- All configuration methods on ``BenchmarkPlotter`` return ``self`` to support
  fluent chaining.
- The plotter validates that all required columns exist in every benchmark
  table and that the axes mapping contains a key for each registered metric
  before delegating to ``draw``.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.visualization.abstracts import IMetricVisualizer
from pysatl_cpd.analysis.visualization.benchmarking.metrics import (
    ARLBasedMetricVisualizer,
    PrAucVisualizer,
    ThresholdBasedMetricVisualizer,
)
from pysatl_cpd.analysis.visualization.benchmarking.plotters.benchmark_plotter import (
    BenchmarkPlotter,
    MetricPlotName,
    MetricVisualizerName,
)

__all__ = [
    "BenchmarkPlotter",
    "IMetricVisualizer",
    "MetricVisualizerName",
    "MetricPlotName",
    "PrAucVisualizer",
    "ThresholdBasedMetricVisualizer",
    "ARLBasedMetricVisualizer",
]
