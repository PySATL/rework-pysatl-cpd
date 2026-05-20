# -*- coding: ascii -*-
"""Abstract benchmark metric visualizer."""

from __future__ import annotations

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import abstractmethod
from collections.abc import Hashable
from typing import Self, Unpack

import pandas as pd

from pysatl_cpd.analysis.visualization.abstracts.ivisualizer import IVisualizer
from pysatl_cpd.analysis.visualization.specs import LineSpec
from pysatl_cpd.analysis.visualization.typedefs import (
    AxMapping,
    DrawBackend,
    GoAxes,
    GoAxMapping,
    GoFigure,
    PltAxes,
    PltAxMapping,
    PltFigure,
)

METRIC_AXIS_NAME = "metric"
type BenchmarkEntryKey = Hashable


class IMetricVisualizer(IVisualizer):
    """
    Base class for benchmark metric visualizers.

    A metric visualizer draws a single benchmark result subplot and can be
    composed by plotters through the shared ``IVisualizer`` interface.
    """

    def __init__(self, backend: DrawBackend | str = DrawBackend.MATPLOTLIB) -> None:
        super().__init__(backend)
        self._benchmark_tables: dict[BenchmarkEntryKey, pd.DataFrame] | None = None
        self._entry_style_map: dict[object, LineSpec] = {}
        self._entry_metric_style_map: dict[object, dict[str, LineSpec]] = {}

    @property
    def axes(self) -> set[str]:
        """
        Return required axis names.

        Returns
        -------
        axes
            Single-axis requirement used by benchmark plotters.
        """
        return {METRIC_AXIS_NAME}

    @property
    @abstractmethod
    def requirements(self) -> list[str]:  # pragma: no cover
        """
        Return required benchmark table columns.

        Returns
        -------
        requirements
            Required column names.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this property.
        """
        raise NotImplementedError

    def set_benchmark_tables(self, benchmark_tables: dict[BenchmarkEntryKey, pd.DataFrame]) -> Self:
        """
        Store benchmark tables keyed by algorithm name.

        Parameters
        ----------
        benchmark_tables
            Benchmark result tables keyed by algorithm name.

        Returns
        -------
        visualizer
            Current visualizer for method chaining.
        """
        self._benchmark_tables = benchmark_tables
        return self

    def set_entry_draw_opts(self, *, entry: BenchmarkEntryKey, **options: Unpack[LineSpec]) -> Self:
        """Set line draw options for all metrics of one benchmark entry."""
        style = self._entry_style_map.setdefault(entry, {})
        style.update(options)
        return self

    def set_entry_metric_draw_opts(
        self,
        *,
        entry: BenchmarkEntryKey,
        metric: str,
        **options: Unpack[LineSpec],
    ) -> Self:
        """Set line draw options for one metric of one benchmark entry."""
        metric_style_map = self._entry_metric_style_map.setdefault(entry, {})
        style = metric_style_map.setdefault(metric, {})
        style.update(options)
        return self

    def draw(self, *, figure: GoFigure | PltFigure, axes: AxMapping) -> GoFigure | PltFigure:
        """
        Draw the metric on provided axes.

        Parameters
        ----------
        figure
            Target figure.
        axes
            ``IVisualizer`` axes mapping containing the ``metric`` axis.

        Returns
        -------
        figure
            Figure with the metric visualizers drawn.
        """
        self._validate_required_columns()
        self._get_metric_axis(axes)
        return super().draw(figure=figure, axes=axes)

    def _get_metric_axis(self, axes: AxMapping) -> GoAxes | PltAxes:
        """Return the metric axis from the required axes mapping.

        Parameters
        ----------
        axes
            Mapping from axis names to axes.

        Returns
        -------
        GoAxes | PltAxes
            The resolved metric axis.

        Raises
        ------
        ValueError
            If the mapping does not contain the ``metric`` key.
        """
        if METRIC_AXIS_NAME not in axes:
            raise ValueError(f"Axes mapping does not contain key: {METRIC_AXIS_NAME}")
        return axes[METRIC_AXIS_NAME]

    @abstractmethod
    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxMapping) -> PltFigure:  # pragma: no cover
        """Draw a metric using a Matplotlib axes mapping."""
        raise NotImplementedError

    @abstractmethod
    def _draw_plotly(self, figure: GoFigure, axes: GoAxMapping) -> GoFigure:  # pragma: no cover
        """Draw a metric using a Plotly axes mapping."""
        raise NotImplementedError

    def _require_tables(self) -> dict[BenchmarkEntryKey, pd.DataFrame]:
        """Return the stored benchmark tables, raising if unset or empty.

        Returns
        -------
        dict[str, pd.DataFrame]
            Tables keyed by algorithm name.

        Raises
        ------
        ValueError
            If tables were not set or are empty.
        """
        if self._benchmark_tables is None:
            raise ValueError("Benchmark tables are not set.")
        if not self._benchmark_tables:
            raise ValueError("Benchmark tables are empty.")
        return self._benchmark_tables

    def _iter_algorithm_tables(self) -> list[tuple[BenchmarkEntryKey, pd.DataFrame]]:
        """Iterate over (entry key, table) pairs from stored tables.

        Returns
        -------
        list[tuple[BenchmarkEntryKey, pd.DataFrame]]
            List of entry-table pairs.
        """
        return list(self._require_tables().items())

    def _resolve_entry_style[T](self, style_map: dict[object, T], entry_key: BenchmarkEntryKey) -> T | None:
        """Resolve a style/config entry by exact key or string fallback.

        Parameters
        ----------
        style_map
            Mapping of user-provided entry selectors to style/config values.
        entry_key
            Benchmark entry key from the stored tables.

        Returns
        -------
        T or None
            Matching value when found, otherwise None.
        """
        if entry_key in style_map:
            return style_map[entry_key]

        entry_label = str(entry_key)
        if entry_label in style_map:
            return style_map[entry_label]

        return None

    def _resolve_line_style(
        self,
        *,
        metric_style_map: dict[str, LineSpec],
        entry_key: BenchmarkEntryKey,
        metric: str,
    ) -> LineSpec:
        """Resolve style for an entry-metric pair preserving old API precedence."""
        style: LineSpec = {}

        entry_style = self._resolve_entry_style(self._entry_style_map, entry_key)
        if entry_style is not None:
            style.update(entry_style)

        metric_style = metric_style_map.get(metric)
        if metric_style is not None:
            style.update(metric_style)

        entry_metric_styles = self._resolve_entry_style(self._entry_metric_style_map, entry_key)
        if entry_metric_styles is not None:
            entry_metric_style = entry_metric_styles.get(metric)
            if entry_metric_style is not None:
                style.update(entry_metric_style)

        return style

    def _validate_required_columns(self) -> None:
        """Check that all required columns exist in every algorithm table.

        Raises
        ------
        ValueError
            If any required column is missing from any algorithm table.
        """
        required_columns = self.requirements
        missing_by_algorithm: dict[BenchmarkEntryKey, list[str]] = {}
        for algorithm_name, table in self._iter_algorithm_tables():
            missing = [column for column in required_columns if column not in table.columns]
            if missing:
                missing_by_algorithm[algorithm_name] = missing
        if missing_by_algorithm:
            raise ValueError(f"Missing required columns for {self.__class__.__name__}: {missing_by_algorithm}")
