# -*- coding: ascii -*-
"""
Threshold-based metric visualizer.
"""

from __future__ import annotations

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Literal

import pandas as pd
import plotly.graph_objects as go

from pysatl_cpd.analysis.visualization.abstracts import IMetricVisualizer
from pysatl_cpd.analysis.visualization.abstracts.imetric_visualizer import METRIC_AXIS_NAME
from pysatl_cpd.analysis.visualization.specs import LineSpec
from pysatl_cpd.analysis.visualization.typedefs import DrawBackend, GoAxMapping, GoFigure, PltAxMapping, PltFigure
from pysatl_cpd.analysis.visualization.utils import line_spec_to_plotly_trace_kwargs

PrecisionMode = Literal["default", "monotonic", "both"]
PLOTLY_GRID_WIDTH = 1
PLOTLY_GRID_COLOR = "lightgray"
MPL_GRID_ALPHA = 0.3


class ThresholdBasedMetricVisualizer(IMetricVisualizer):
    """
    Draw metrics as functions of threshold.
    """

    def __init__(
        self,
        *,
        backend: DrawBackend | str = DrawBackend.MATPLOTLIB,
        y_metrics: list[str],
        title: str | None = None,
        ylabel: str | None = None,
        precision_mode: PrecisionMode = "default",
        style_map: dict[str, LineSpec] | None = None,
    ) -> None:
        super().__init__(backend=backend)
        self._y_metrics = y_metrics
        self._title = title or "Threshold-based metrics"
        self._ylabel = ylabel or "Value"
        self._precision_mode = precision_mode
        self._style_map = style_map or {}

    @property
    def requirements(self) -> list[str]:
        """Return required columns: threshold plus all configured Y metrics."""
        columns = ["threshold", *self._y_metrics]
        return list(dict.fromkeys(columns))

    # def _recalculate_stats(self, table: pd.DataFrame) -> pd.DataFrame:
    #     result = table.sort_values(by="threshold", ascending=True).copy()
    #     if "precision" in result.columns:
    #         result["precision"] = result["precision"].cummax()
    #     if "f1" in result.columns and "precision" in result.columns and "recall" in result.columns:
    #         precision = result["precision"]
    #         recall = result["recall"]
    #         result["f1"] = (2 * precision * recall) / (precision + recall).replace(0, pd.NA)
    #     return result
    # TODO: Add new column for monotonic precision outside of the visualizer by overloading
    def _recalculate_stats(self, table: pd.DataFrame) -> pd.DataFrame:
        """Apply monotonic precision and recompute F1 from it.

        Parameters
        ----------
        table
            Benchmark table sorted by threshold.

        Returns
        -------
        pd.DataFrame
            Table with cumulatively maximized precision and recomputed F1.
        """
        result = table.sort_values(by="threshold", ascending=True).copy()
        if "precision" in result.columns:
            result["precision"] = result["precision"].cummax()
        if "f1" in result.columns and "precision" in result.columns and "recall" in result.columns:
            precision = result["precision"]
            recall = result["recall"]
            result["f1"] = (2 * precision * recall) / (precision + recall).replace(0, pd.NA)
        return result

    def _iter_mode_tables(self, table: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
        """Iterate over precision mode variants (default, monotonic, both).

        Parameters
        ----------
        table
            Benchmark table to split into mode variants.

        Returns
        -------
        list[tuple[str, pd.DataFrame]]
            List of (mode_name, mode_table) tuples.

        Raises
        ------
        ValueError
            If ``precision_mode`` is not one of ``default``, ``monotonic``, or ``both``.
        """
        if self._precision_mode not in {"default", "monotonic", "both"}:
            raise ValueError(f"Unsupported precision mode: {self._precision_mode}")
        table = table.sort_values(by="threshold", ascending=True)
        mode_tables: list[tuple[str, pd.DataFrame]] = []
        if self._precision_mode in {"default", "both"}:
            mode_tables.append(("default", table))
        if self._precision_mode in {"monotonic", "both"}:
            mode_tables.append(("monotonic", self._recalculate_stats(table)))
        return mode_tables

    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxMapping) -> PltFigure:
        """Draw threshold-based metric curves on a Matplotlib axes.

        Parameters
        ----------
        figure
            Matplotlib figure.
        axes
            Matplotlib axes mapping containing the metric axes.

        Returns
        -------
        pltFigure
            The figure with curves drawn.
        """
        ax = axes[METRIC_AXIS_NAME]
        for entry_key, table in self._iter_algorithm_tables():
            mode_tables = self._iter_mode_tables(table)
            for mode, mode_table in mode_tables:
                mode_suffix = "" if mode == "default" else " (monotonic)"
                for metric in self._y_metrics:
                    style = self._resolve_line_style(
                        metric_style_map=self._style_map, entry_key=entry_key, metric=metric
                    )
                    ax.plot(
                        mode_table["threshold"],
                        mode_table[metric],
                        label=f"{entry_key}: {metric}{mode_suffix}",
                        linestyle=style.get("linestyle", "--" if mode == "monotonic" else "-"),
                        color=style.get("color"),
                        linewidth=float(style.get("linewidth", 1)),
                    )
        ax.set_title(self._title)
        ax.set_xlabel("Threshold")
        ax.set_ylabel(self._ylabel)
        ax.grid(True, alpha=MPL_GRID_ALPHA)
        ax.legend(loc="best")
        return figure

    def _draw_plotly(self, figure: GoFigure, axes: GoAxMapping) -> GoFigure:
        """Draw threshold-based metric curves on a Plotly subplot.

        Parameters
        ----------
        figure
            Plotly figure.
        axes
            Plotly axes mapping containing the metric subplot position.

        Returns
        -------
        GoFigure
            The figure with curves drawn.
        """
        row, col = axes[METRIC_AXIS_NAME]
        for entry_key, table in self._iter_algorithm_tables():
            mode_tables = self._iter_mode_tables(table)
            for mode, mode_table in mode_tables:
                mode_suffix = "" if mode == "default" else " (monotonic)"
                for metric in self._y_metrics:
                    style = self._resolve_line_style(
                        metric_style_map=self._style_map, entry_key=entry_key, metric=metric
                    )
                    effective_style: LineSpec = {"linestyle": "--" if mode == "monotonic" else "-", "linewidth": 1}
                    effective_style.update(style)
                    figure.add_trace(
                        go.Scatter(
                            x=mode_table["threshold"],
                            y=mode_table[metric],
                            mode="lines",
                            name=f"{entry_key}: {metric}{mode_suffix}",
                            **line_spec_to_plotly_trace_kwargs(effective_style),
                        ),
                        row=row,
                        col=col,
                    )
        figure.update_xaxes(
            title_text="Threshold",
            showgrid=True,
            gridwidth=PLOTLY_GRID_WIDTH,
            gridcolor=PLOTLY_GRID_COLOR,
            row=row,
            col=col,
        )
        figure.update_yaxes(
            title_text=self._ylabel,
            showgrid=True,
            gridwidth=PLOTLY_GRID_WIDTH,
            gridcolor=PLOTLY_GRID_COLOR,
            row=row,
            col=col,
        )
        return figure
