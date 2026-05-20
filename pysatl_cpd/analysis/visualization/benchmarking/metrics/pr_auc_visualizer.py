# -*- coding: ascii -*-
"""
Precision-Recall AUC visualizer.
"""

from __future__ import annotations

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pysatl_cpd.analysis.visualization.abstracts import IMetricVisualizer
from pysatl_cpd.analysis.visualization.abstracts.imetric_visualizer import (
    METRIC_AXIS_NAME,
)
from pysatl_cpd.analysis.visualization.specs import LineSpec
from pysatl_cpd.analysis.visualization.typedefs import (
    DrawBackend,
    GoAxMapping,
    GoFigure,
    PltAxMapping,
    PltFigure,
)
from pysatl_cpd.analysis.visualization.utils import line_spec_to_plotly_trace_kwargs

PLOTLY_GRID_WIDTH = 1
PLOTLY_GRID_COLOR = "lightgray"
MPL_GRID_ALPHA = 0.3


class PrAucVisualizer(IMetricVisualizer):
    """
    Draw precision-recall curve and PR-AUC value.
    """

    def __init__(
        self,
        *,
        backend: DrawBackend | str = DrawBackend.MATPLOTLIB,
        label: str = "PR-AUC",
        color: str | None = None,
        linestyle: str = "-",
        marker: str = "o",
        markersize: float | None = None,
    ) -> None:
        super().__init__(backend=backend)
        self._label = label
        self._style_map: dict[str, LineSpec] = {label: {"linestyle": linestyle, "marker": marker}}
        if color is not None:
            self._style_map[label]["color"] = color
        if markersize is not None:
            self._style_map[label]["markersize"] = markersize

    @property
    def requirements(self) -> list[str]:
        """Required columns: recall and precision.

        Returns
        -------
        list[str]
        """
        return ["recall", "precision"]

    # TODO: Move to self._benchmark_table property
    def _prepare_pr_data(self, table: pd.DataFrame) -> pd.DataFrame:
        """Sort and deduplicate precision-recall data, adding boundary points.

        Parameters
        ----------
        table
            Benchmark table containing ``recall`` and ``precision`` columns.

        Returns
        -------
        pd.DataFrame
            Sorted PR data with boundary points appended.
        """
        pr_data = (
            table[["recall", "precision"]]
            .sort_values(by=["recall", "precision"], ascending=[True, False])
            .drop_duplicates(subset=["recall"], keep="first")
        )
        # NOTE: Need to check if this is correct
        boundary_points = pd.DataFrame([{"recall": 0.0, "precision": 1.0}, {"recall": 1.0, "precision": 0.0}])
        return pd.concat([pr_data, boundary_points], ignore_index=True).sort_values(by="recall")

    # TODO: MOve outside of the visualizer. Pass to label
    def _compute_auc(self, pr_data: pd.DataFrame) -> float:
        """Compute the area under the precision-recall curve.

        Parameters
        ----------
        pr_data
            Prepared PR data with ``recall`` and ``precision`` columns.

        Returns
        -------
        float
            The PR-AUC score.
        """
        return float(np.trapezoid(pr_data["precision"], pr_data["recall"]))

    # TODO: TypeDict for options. Add special setter for style map.
    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxMapping) -> PltFigure:
        """Draw precision-recall curves on a Matplotlib axes.

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
            pr_data = self._prepare_pr_data(table)
            auc_score = self._compute_auc(pr_data)
            style = self._resolve_line_style(metric_style_map=self._style_map, entry_key=entry_key, metric=self._label)
            ax.plot(
                pr_data["recall"],
                pr_data["precision"],
                marker=style.get("marker", "o"),
                label=f"{entry_key}: {self._label} (AUC = {auc_score:.3f})",
                linestyle=style.get("linestyle", "-"),
                color=style.get("color"),
                linewidth=float(style.get("linewidth", 1)),
                markersize=float(style["markersize"]) if "markersize" in style else None,
            )
        ax.set_title("PR-AUC")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=MPL_GRID_ALPHA)
        ax.legend(loc="best")
        return figure

    def _draw_plotly(self, figure: GoFigure, axes: GoAxMapping) -> GoFigure:
        """Draw precision-recall curves on a Plotly subplot.

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
            pr_data = self._prepare_pr_data(table)
            auc_score = self._compute_auc(pr_data)
            style = self._resolve_line_style(metric_style_map=self._style_map, entry_key=entry_key, metric=self._label)
            figure.add_trace(
                go.Scatter(
                    x=pr_data["recall"],
                    y=pr_data["precision"],
                    mode="lines+markers",
                    name=f"{entry_key}: {self._label} (AUC = {auc_score:.3f})",
                    **line_spec_to_plotly_trace_kwargs(style),
                ),
                row=row,
                col=col,
            )
        figure.update_xaxes(
            title_text="Recall",
            range=[0.0, 1.05],
            showgrid=True,
            gridwidth=PLOTLY_GRID_WIDTH,
            gridcolor=PLOTLY_GRID_COLOR,
            row=row,
            col=col,
        )
        figure.update_yaxes(
            title_text="Precision",
            range=[0.0, 1.05],
            showgrid=True,
            gridwidth=PLOTLY_GRID_WIDTH,
            gridcolor=PLOTLY_GRID_COLOR,
            row=row,
            col=col,
        )
        figure.update_layout(showlegend=True)
        return figure
