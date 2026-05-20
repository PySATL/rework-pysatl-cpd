# -*- coding: ascii -*-
"""
Metric visualizers for benchmark results.

This module provides concrete implementations of ``IMetricVisualizer`` that
render benchmark metric tables as publication-ready plots. Each visualizer
consumes a pandas DataFrame containing precomputed metrics and draws one or
more curves on a shared subplot. All three classes support both Matplotlib
and Plotly backends through the ``DrawBackend`` enum.

.. raw:: html

    <h2>Public API</h2>

- ``ARLBasedMetricVisualizer`` -- plots metrics as functions of average run
  length (ARL). Rows sharing the same ARL value are collapsed to their
  minimum, producing a lower-envelope curve.
- ``PrAucVisualizer`` -- draws precision-recall curves and annotates each
  curve with its computed PR-AUC score.
- ``ThresholdBasedMetricVisualizer`` -- plots metrics as functions of
  detection threshold. Supports an optional monotonic-precision mode that
  applies cumulative-max precision and recomputes F1 accordingly.

Each visualizer is designed to be composed inside a ``BenchmarkPlotter``,
which manages the overall figure layout and delegates to individual
visualizers for each subplot. See the ``plotters`` subpackage for details.

.. raw:: html

    <h2>Examples</h2>

Threshold-based metric curves with Matplotlib::

    >>> import matplotlib.pyplot as plt
    >>> import pandas as pd
    >>> from pysatl_cpd.analysis.visualization.benchmarking.metrics import (
    ...     ThresholdBasedMetricVisualizer,
    ... )
    >>> table = pd.DataFrame({
    ...     "threshold": [0.5, 1.0, 1.5, 2.0],
    ...     "precision": [0.2, 0.5, 0.8, 0.95],
    ...     "recall": [0.95, 0.85, 0.6, 0.3],
    ... })
    >>> visualizer = ThresholdBasedMetricVisualizer(
    ...     y_metrics=["precision", "recall"],
    ...     title="Precision and Recall",
    ...     ylabel="Value",
    ... )
    >>> visualizer.set_benchmark_tables({"algo": table})
    >>> fig, ax = plt.subplots()
    >>> visualizer.draw(figure=fig, axes={"metric": ax})

ARL-based envelope curves with Plotly::

    >>> import pandas as pd
    >>> from plotly.subplots import make_subplots
    >>> from pysatl_cpd.analysis.visualization import DrawBackend
    >>> from pysatl_cpd.analysis.visualization.benchmarking.metrics import (
    ...     ARLBasedMetricVisualizer,
    ... )
    >>> table = pd.DataFrame({
    ...     "arl": [100, 120, 140, 160],
    ...     "f1": [0.4, 0.6, 0.75, 0.85],
    ... })
    >>> visualizer = ARLBasedMetricVisualizer(
    ...     backend=DrawBackend.PLOTLY,
    ...     y_metrics=["f1"],
    ...     title="F1 vs ARL",
    ...     ylabel="F1",
    ... )
    >>> visualizer.set_benchmark_tables({"algo": table})
    >>> fig = make_subplots(rows=1, cols=1)
    >>> visualizer.draw(figure=fig, axes={"metric": (1, 1)})

Precision-recall curve with PR-AUC annotation::

    >>> import matplotlib.pyplot as plt
    >>> import pandas as pd
    >>> from pysatl_cpd.analysis.visualization.benchmarking.metrics import (
    ...     PrAucVisualizer,
    ... )
    >>> table = pd.DataFrame({
    ...     "recall": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    ...     "precision": [1.0, 0.9, 0.8, 0.6, 0.3, 0.0],
    ... })
    >>> visualizer = PrAucVisualizer(label="PR-AUC", color="blue")
    >>> visualizer.set_benchmark_tables({"algo": table})
    >>> fig, ax = plt.subplots()
    >>> visualizer.draw(figure=fig, axes={"metric": ax})

.. raw:: html

    <h2>Notes</h2>

- All visualizers inherit from ``IMetricVisualizer`` and follow the same
  lifecycle: instantiate with configuration, call ``set_benchmark_tables``
  with a dict of ``{entry_key: DataFrame}``, then ``draw`` onto a figure
  and axes mapping that includes the ``"metric"`` key.
- ``ARLBasedMetricVisualizer`` computes a lower envelope by taking the
  group-wise minimum of each Y metric for rows that share the same ARL
  value. This is useful when multiple thresholds produce identical ARLs.
- ``ThresholdBasedMetricVisualizer`` offers a ``precision_mode`` parameter
  (``"default"``, ``"monotonic"``, or ``"both"``). The monotonic mode
  applies cumulative-max to precision and recomputes F1 from the adjusted
  values.
- ``PrAucVisualizer`` computes the area under the precision-recall curve
  using ``numpy.trapezoid`` and appends boundary points ``(0, 1)`` and
  ``(1, 0)`` before integration.
- Required third-party packages: ``pandas``, ``numpy``, ``matplotlib``,
  and ``plotly``.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.visualization.benchmarking.metrics.arl_based_metric_visualizer import (
    ARLBasedMetricVisualizer,
)
from pysatl_cpd.analysis.visualization.benchmarking.metrics.pr_auc_visualizer import (
    PrAucVisualizer,
)
from pysatl_cpd.analysis.visualization.benchmarking.metrics.threshold_based_metric_visualizer import (
    ThresholdBasedMetricVisualizer,
)

__all__ = [
    "PrAucVisualizer",
    "ThresholdBasedMetricVisualizer",
    "ARLBasedMetricVisualizer",
]
