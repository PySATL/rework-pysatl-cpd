# -*- coding: ascii -*-
"""
Benchmark plotter coordinator.
"""

from __future__ import annotations

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Hashable
from typing import Self, Unpack, cast

import pandas as pd

from pysatl_cpd.analysis.visualization.abstracts import IMetricVisualizer
from pysatl_cpd.analysis.visualization.abstracts.imetric_visualizer import METRIC_AXIS_NAME
from pysatl_cpd.analysis.visualization.specs import LineSpec
from pysatl_cpd.analysis.visualization.typedefs import Axes, AxMapping, Figure

type MetricVisualizerName = str
type MetricPlotName = str


class BenchmarkPlotter:
    """
    Coordinate benchmark metric visualizers.
    """

    def __init__(self) -> None:
        self._benchmark_tables: dict[Hashable, pd.DataFrame] | None = None
        self._metrics: dict[MetricVisualizerName, IMetricVisualizer] = {}

    @property
    def requirements(self) -> list[str]:
        """Return the union of all required columns from registered metrics."""
        all_requirements: list[str] = []
        for metric in self._metrics.values():
            all_requirements.extend(metric.requirements)
        return list(dict.fromkeys(all_requirements))

    def __getitem__(self, metric_name: MetricVisualizerName) -> IMetricVisualizer:
        """Access a registered metric visualizer by name.

        Parameters
        ----------
        metric_name
            Name of the metric to retrieve.

        Returns
        -------
        IMetricVisualizer
            The registered metric visualizer.

        Raises
        ------
        KeyError
            If the metric name is not registered.
        """
        try:
            return self._metrics[metric_name]
        except KeyError as exc:
            raise KeyError(f"Metric visualizer '{metric_name}' is not registered.") from exc

    def set_benchmark_tables(self, benchmark_tables: dict[Hashable, pd.DataFrame]) -> Self:
        """Set benchmark result tables and propagate them to all metrics.

        Parameters
        ----------
        benchmark_tables
            Benchmark result tables keyed by algorithm name.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._benchmark_tables = benchmark_tables
        for metric in self._metrics.values():
            metric.set_benchmark_tables(benchmark_tables)
        return self

    def set_metrics(self, metrics: dict[MetricVisualizerName, IMetricVisualizer]) -> Self:
        """Register metric visualizers and optionally propagate tables.

        Parameters
        ----------
        metrics
            Named metric visualizers to register.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._metrics = metrics
        if self._benchmark_tables is not None:
            for metric in self._metrics.values():
                metric.set_benchmark_tables(self._benchmark_tables)
        return self

    def set_entry_draw_opts(self, *, entry: Hashable, **options: Unpack[LineSpec]) -> Self:
        """Set entry-wide line draw options on all registered metric visualizers."""
        for metric in self._metrics.values():
            metric.set_entry_draw_opts(entry=entry, **options)
        return self

    def set_entry_metric_draw_opts(self, *, entry: Hashable, metric: str, **options: Unpack[LineSpec]) -> Self:
        """Set entry+metric line draw options on all registered metric visualizers."""
        for visualizer in self._metrics.values():
            visualizer.set_entry_metric_draw_opts(entry=entry, metric=metric, **options)
        return self

    def draw(self, figure: Figure, axes: dict[MetricVisualizerName, Axes]) -> Figure:
        """Coordinate drawing of all registered metric visualizers.

        Parameters
        ----------
        figure
            The figure to draw on.
        axes
            Mapping from metric names to their axes.

        Returns
        -------
        Figure
            The figure with all metric visuals drawn.

        Raises
        ------
        ValueError
            If benchmark tables or metrics are not set, or required columns
            or axes are missing.
        """
        if self._benchmark_tables is None:
            raise ValueError("Benchmark tables are not set.")
        if not self._benchmark_tables:
            raise ValueError("Benchmark tables are empty.")
        if not self._metrics:
            raise ValueError("Metrics are not set.")

        missing_columns_by_algorithm: dict[Hashable, list[str]] = {}
        for algorithm_name, benchmark_table in self._benchmark_tables.items():
            missing_columns = [column for column in self.requirements if column not in benchmark_table.columns]
            if missing_columns:
                missing_columns_by_algorithm[algorithm_name] = missing_columns
        if missing_columns_by_algorithm:
            raise ValueError(f"Missing required columns for BenchmarkPlotter: {missing_columns_by_algorithm}")

        missing_axes = [metric_name for metric_name in self._metrics if metric_name not in axes]
        if missing_axes:
            raise ValueError(f"Axes mapping does not contain keys: {missing_axes}")

        for metric_name, metric in self._metrics.items():
            metric_axes = cast(AxMapping, {METRIC_AXIS_NAME: axes[metric_name]})
            figure = metric.draw(figure=figure, axes=metric_axes)
        return figure
