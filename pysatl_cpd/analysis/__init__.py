# -*- coding: ascii -*-
"""Analysis module for change-point detection evaluation and visualization.

This module provides a unified surface for evaluating and visualizing
change-point detection (CPD) algorithms. It combines two complementary
subpackages: one for quantitative evaluation metrics and one for rendering
detection results as publication-ready figures.

.. raw:: html

    <h2>Public API</h2>

Abstract metric interfaces:

- ``ISingleRunMetric`` -- base class for metrics evaluated on one
  ``SingleRun``. See :mod:`pysatl_cpd.analysis.metrics.abstracts`.
- ``IMultipleRunMetric`` -- base class for metrics evaluated over a
  sequence of ``SingleRun`` objects. See
  :mod:`pysatl_cpd.analysis.metrics.abstracts`.

Single-run metrics (operate on one ``SingleRun``):

- ``ClassificationPrimitive`` -- base class for count-based classification
  metrics. See :mod:`pysatl_cpd.analysis.metrics.single_run`.
- ``TruePositiveCount`` -- counts true change points with matched detections.
- ``FalsePositiveCount`` -- counts unmatched detections.
- ``FalseNegativeCount`` -- counts true change points with no detection.
- ``Delays`` -- per-change-point detection delays for online algorithms.
- ``RunLengths`` -- distances between consecutive detections.

Multiple-run aggregation metrics (operate on a sequence of ``SingleRun``):

- ``AggregationMetric`` -- abstract base that reduces per-run results via a
  user-defined method. See :mod:`pysatl_cpd.analysis.metrics.multiple_run`.
- ``TotalSum`` -- sums per-run numeric results.
- ``TotalMean`` -- arithmetic mean of per-run numeric results.
- ``TotalMedian`` -- median of per-run numeric results.
- ``DerivedMetric`` -- combines multiple multi-run metric outputs.

Multiple-run classification metrics:

- ``TotalTP`` -- total true positives across all runs.
- ``TotalFP`` -- total false positives across all runs.
- ``TotalFN`` -- total false negatives across all runs.
- ``PrecisionMetric`` -- micro-averaged precision.
- ``RecallMetric`` -- micro-averaged recall.
- ``FScoreMetric`` -- F-beta score (F1 when ``beta=1``).
- ``ClassificationReport`` -- full classification summary dict.

Multiple-run online metrics:

- ``ARLMetric`` -- mean average run length across all runs.
- ``MeanDelayMetric`` -- mean detection delay across all runs.
- ``MedianDelayMetric`` -- median detection delay across all runs.

Abstract visualizer interfaces:

- ``IVisualizer`` -- base interface for visualizers that manage complete
  subplots. See :mod:`pysatl_cpd.analysis.visualization.abstracts`.
- ``ITimeseriesVisualizer`` -- interface for time series visualizers.
- ``ITraceVisualizer`` -- interface for detection trace visualizers.
- ``IMetricVisualizer`` -- base class for benchmark metric visualizers.
- ``IVisualComponent`` -- interface for composable drawing components.

Time series visualizers:

- ``UnivariateTimeseriesVisualizer`` -- renders a single time series.
- ``PlainMultivariateTimeseriesVisualizer`` -- dimension-oriented visualizer
  with one subplot per data dimension.
- ``RichMultivariateTimeseriesVisualizer`` -- pandas-first visualizer with
  logical plot names and twin y-axis support.
- ``MultivariateTimeseriesVisualizer`` -- compatibility alias for
  ``PlainMultivariateTimeseriesVisualizer``.

Online detection visualizers:

- ``OnlineTraceVisualizer`` -- renders detection function, threshold, and
  processing time panels. Accepts a composable state visualizer for
  algorithm-specific internal state panels.
- ``IOnlineStateVisualizer`` -- abstract interface for algorithm state
  visualizers.
- ``DummyStateVisualizer`` -- placeholder state visualizer for testing.

Online CPD plotter:

- ``OnlineCpdPlotter`` -- high-level coordinator that bundles a timeseries
  visualizer, a trace visualizer, and annotation components into one draw
  call. Supports named layout strings (``"vertical"``, ``"split"``,
  ``"dashboard-lite"``, ``"dashboard"``).

Benchmark visualization:

- ``BenchmarkPlotter`` -- coordinates multiple metric visualizers to produce
  composite benchmark figures.
- ``ThresholdBasedMetricVisualizer`` -- plots metrics as functions of
  detection threshold.
- ``ARLBasedMetricVisualizer`` -- plots metrics as functions of average run
  length with lower-envelope curves.
- ``PrAucVisualizer`` -- draws precision-recall curves with PR-AUC annotation.

Reusable components and specifications:

- ``VerticalLineVisualComponent`` -- draws vertical lines at specified
  x-coordinates.
- ``VerticalFillComponent`` -- draws filled vertical regions between pairs
  of x-coordinates.
- ``DrawBackend`` -- enum selecting ``MATPLOTLIB`` or ``PLOTLY`` rendering.
- ``PlotSpec`` -- TypedDict for subplot-level options (title, axis labels).
- ``LineSpec`` -- TypedDict for line-style options.
- ``FilledLineSpec`` -- TypedDict extending ``LineSpec`` with fill options.
- ``FillSpec`` -- TypedDict for fill-style options.

Utilities:

- ``get_plotly_subplot_annotation_index`` -- helper for Plotly annotation
  indexing.

.. raw:: html

    <h2>Subpackages</h2>

- ``metrics`` -- evaluation metrics for CPD algorithms, organized into
  single-run and multiple-run scopes. See the subpackage docstring for
  the full API, examples, and notes.
- ``visualization`` -- rendering layer for time series, detection traces,
  algorithm states, and benchmark results. Supports both Matplotlib and
  Plotly backends. See the subpackage docstring for the full API and
  composition patterns.

Notes
-----
- Requires ``matplotlib`` and ``plotly`` for visualization functionality.
- Requires ``pandas`` for benchmark visualizers and rich multivariate
  visualizers.
- Classification metrics use an ``error_margin`` tuple ``(left, right)``
  to define a tolerance window around each true change point for matching
  detections.
- Change-point indices are zero-based throughout the analysis module.
- All visualizers and components are backend-agnostic; the same conceptual
  figure can be rendered with either Matplotlib or Plotly by switching the
  ``DrawBackend`` enum value.
- The module requires Python 3.12+ for PEP 695 generic syntax used in
  visualizer type parameters.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics import *
from pysatl_cpd.analysis.metrics import __all__ as _metrics_all
from pysatl_cpd.analysis.visualization import *
from pysatl_cpd.analysis.visualization import __all__ as _visualization_all

__all__ = [
    *_metrics_all,
    *_visualization_all,
]

del _metrics_all
del _visualization_all
