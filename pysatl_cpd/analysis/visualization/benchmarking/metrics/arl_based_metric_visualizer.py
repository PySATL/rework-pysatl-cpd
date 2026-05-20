# -*- coding: ascii -*-
"""
ARL-based metric visualizer.
"""

from __future__ import annotations

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pandas as pd
import plotly.graph_objects as go

from pysatl_cpd.analysis.visualization.abstracts import IMetricVisualizer
from pysatl_cpd.analysis.visualization.abstracts.imetric_visualizer import METRIC_AXIS_NAME
from pysatl_cpd.analysis.visualization.specs import LineSpec
from pysatl_cpd.analysis.visualization.typedefs import DrawBackend, GoAxMapping, GoFigure, PltAxMapping, PltFigure
from pysatl_cpd.analysis.visualization.utils import line_spec_to_plotly_trace_kwargs

PLOTLY_GRID_WIDTH = 1
PLOTLY_GRID_COLOR = "lightgray"
MPL_GRID_ALPHA = 0.3


class ARLBasedMetricVisualizer(IMetricVisualizer):
    """
    Draw metrics as functions of ARL.

    Rows that share the same ``arl`` value (e.g. different thresholds) are
    collapsed by taking the minimum of each plotted metric, yielding the
    lower envelope along the ARL axis.
    """

    def __init__(
        self,
        *,
        backend: DrawBackend | str = DrawBackend.MATPLOTLIB,
        y_metrics: list[str],
        title: str = "ARL Curve",
        ylabel: str = "Values",
        style_map: dict[str, LineSpec] | None = None,
    ) -> None:
        super().__init__(backend=backend)
        self._y_metrics = y_metrics
        self._title = title
        self._ylabel = ylabel
        self._style_map = style_map or {}

    @property
    def requirements(self) -> list[str]:
        """Return required columns: ARL plus all configured Y metrics."""
        columns = ["arl", *self._y_metrics]
        return list(dict.fromkeys(columns))

    def _arl_metric_envelope(self, table: pd.DataFrame) -> pd.DataFrame:
        """One row per distinct ARL with each metric replaced by its group-wise minimum."""
        metric_cols = list(dict.fromkeys(self._y_metrics))
        return table.groupby("arl", sort=False)[metric_cols].min().reset_index().sort_values("arl", ascending=True)

    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxMapping) -> PltFigure:
        """Draw ARL-based curves on a Matplotlib axes.

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
        for entry_key, benchmark_table in self._iter_algorithm_tables():
            table = self._arl_metric_envelope(benchmark_table)
            for metric in self._y_metrics:
                style = self._resolve_line_style(metric_style_map=self._style_map, entry_key=entry_key, metric=metric)
                ax.plot(
                    table["arl"],
                    table[metric],
                    label=f"{entry_key}: {metric}",
                    linestyle=style.get("linestyle", "-"),
                    color=style.get("color"),
                    linewidth=float(style.get("linewidth", 1)),
                )

        ax.set_title(self._title)
        ax.set_xlabel("ARL")
        ax.set_ylabel(self._ylabel)
        ax.grid(True, alpha=MPL_GRID_ALPHA)
        ax.legend(loc="best")
        return figure

    def _draw_plotly(self, figure: GoFigure, axes: GoAxMapping) -> GoFigure:
        """Draw ARL-based curves on a Plotly subplot.

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
        for entry_key, benchmark_table in self._iter_algorithm_tables():
            table = self._arl_metric_envelope(benchmark_table)
            for metric in self._y_metrics:
                style = self._resolve_line_style(metric_style_map=self._style_map, entry_key=entry_key, metric=metric)
                figure.add_trace(
                    go.Scatter(
                        x=table["arl"],
                        y=table[metric],
                        mode="lines",
                        name=f"{entry_key}: {metric}",
                        **line_spec_to_plotly_trace_kwargs(style),
                    ),
                    row=row,
                    col=col,
                )

        figure.update_xaxes(
            title_text="ARL",
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
