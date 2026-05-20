# -*- coding: ascii -*-
"""
Benchmark plotter coordinator and exports.

This subpackage provides the ``BenchmarkPlotter`` class, which coordinates
multiple metric visualizers to produce composite benchmark figures. The
plotter manages registration of ``IMetricVisualizer`` instances, propagates
benchmark result tables to all registered metrics, and delegates drawing
to each visualizer against a user-supplied axes mapping.

Concrete metric visualizer implementations live in the sibling ``metrics``
subpackage (``ThresholdBasedMetricVisualizer``, ``PrAucVisualizer``, and
``ARLBasedMetricVisualizer``). See that subpackage's docstring for details
on each visualizer's data requirements and rendering behavior.

.. raw:: html

    <h2>Public API</h2>

- ``BenchmarkPlotter``: Coordinates benchmark metric visualizers. Manages
  registration, data propagation, and composite drawing across matplotlib
  and plotly backends.
- ``MetricVisualizerName``: Type alias (``str``) for metric visualizer names
  used as keys in registration and axes mapping.
- ``MetricPlotName``: Type alias (``str``) for metric plot names.

.. raw:: html

    <h2>Examples</h2>

Register metric visualizers and benchmark tables, then draw into a
matplotlib figure::

    >>> import matplotlib.pyplot as plt
    >>> import pandas as pd
    >>> from pysatl_cpd.analysis.visualization.benchmarking.metrics import (
    ...     ThresholdBasedMetricVisualizer,
    ... )
    >>> from pysatl_cpd.analysis.visualization.benchmarking.plotters import (
    ...     BenchmarkPlotter,
    ... )
    >>> benchmark_tables = {
    ...     "shewhart": pd.DataFrame({"threshold": [1.0, 2.0], "f1": [0.8, 0.9]}),
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
    >>> from pysatl_cpd.analysis.visualization.typedefs import DrawBackend
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

.. raw:: html

    <h2>Notes</h2>

- Requires ``pandas``, ``matplotlib``, and ``plotly`` at runtime.
- The plotter validates that all required columns exist in every benchmark
  table and that the axes mapping contains a key for each registered metric
  before delegating to ``draw``.
- All configuration methods return ``self`` to support fluent chaining.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.visualization.benchmarking.plotters.benchmark_plotter import (
    BenchmarkPlotter,
    MetricPlotName,
    MetricVisualizerName,
)

__all__ = [
    "BenchmarkPlotter",
    "MetricVisualizerName",
    "MetricPlotName",
]
