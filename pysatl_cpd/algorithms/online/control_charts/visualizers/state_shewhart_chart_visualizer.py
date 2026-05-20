# -*- coding: ascii -*-
"""
Shewhart state visualizer implementation.

This module provides a visualizer for rendering Shewhart control chart
algorithm state evolution over time, including running mean, control limits,
and sliding window mean.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Self, TypedDict, Unpack

import numpy as np
import plotly.graph_objects as go

from pysatl_cpd.algorithms.online.control_charts.shewhart_control_chart import ShewhartControlChartState
from pysatl_cpd.analysis.visualization.online.states.ionline_state_visualizer import (
    IOnlineStateVisualizer,
)
from pysatl_cpd.analysis.visualization.typedefs import (
    DrawBackend,
    DrawOption,
    GoAxMapping,
    GoFigure,
    PltAxMapping,
    PltFigure,
)
from pysatl_cpd.analysis.visualization.utils import get_plotly_subplot_annotation_index, translate_linestyle

# Plotly constants
PLOTLY_GRID_WIDTH = 1
PLOTLY_GRID_COLOR = "lightgray"

# Matplotlib constants
MPL_GRID_ALPHA = 0.3


class ShewhartStatePlotOpts(TypedDict, total=False):
    """Plot options for Shewhart state subplot."""

    xlabel: str
    ylabel: str
    grid: bool
    title: str | None


class ShewhartStateDrawOpts(TypedDict, total=False):
    """Drawing options for Shewhart state lines."""

    mean_color: str
    mean_linewidth: float
    mean_label: str
    control_limit_color: str
    control_limit_linestyle: str
    control_limit_linewidth: float
    control_limit_label: str
    window_mean_color: str
    window_mean_linewidth: float
    window_mean_label: str
    fill_alpha: float


class ShewhartStateBandOpts(TypedDict, total=False):
    """Band calculation options for Shewhart control limits."""

    band_size: float


class ShewhartStateVisualizer(IOnlineStateVisualizer[ShewhartControlChartState]):
    """
    Visualizer for Shewhart control chart algorithm state evolution.

    This visualizer renders:
    - Running mean (mu)
    - Control limits (mu +/- k * sigma / sqrt(w)) where w is window size
    - Sliding window mean (x_bar_w)

    Parameters
    ----------
    backend
        Plotting backend to use for rendering.
    """

    def __init__(self, backend: DrawBackend) -> None:
        super().__init__(backend)
        self._states: Sequence[ShewhartControlChartState | None] = []

        # Options storage with defaults
        self._plot_opts: ShewhartStatePlotOpts = {
            "xlabel": "Time Index",
            "ylabel": "Value",
            "grid": True,
            "title": None,
        }
        self._draw_opts: ShewhartStateDrawOpts = {
            "mean_color": "blue",
            "mean_linewidth": 1.5,
            "mean_label": "Running Mean (mu)",
            "control_limit_color": "red",
            "control_limit_linestyle": "dash",
            "control_limit_linewidth": 1,
            "control_limit_label": "Control Limits (mu +/- k*sigma/sqrt(w))",
            "window_mean_color": "green",
            "window_mean_linewidth": 1,
            "window_mean_label": "Window Mean (x_bar_w)",
            "fill_alpha": 0.2,
        }
        self._band_opts: ShewhartStateBandOpts = {
            "band_size": 3.0,
        }

    @property
    def axes(self) -> set[str]:
        """
        Declare the subplot names required by this visualizer.

        Returns
        -------
        set[str]
            Set containing "shewhart_state" subplot name.
        """
        return {"shewhart_state"}

    def set_states(
        self,
        states: Sequence[ShewhartControlChartState | None],
        **draw_options: DrawOption,
    ) -> Self:
        """
        Set the sequence of algorithm states to visualize.

        Parameters
        ----------
        states
            Sequence of Shewhart state snapshots for each observation step.
            None values indicate steps where state was not captured.
        **draw_options
            Additional backend-specific drawing options.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._states = states

        # Extract data from states
        self._time_points = np.arange(len(self._states))
        self._means = np.array(
            [
                state.mean if (state is not None and not state.is_in_learning_period) else np.nan
                for state in self._states
            ]
        )
        self._stds = np.array(
            [
                state.standard_deviation if (state is not None and not state.is_in_learning_period) else np.nan
                for state in self._states
            ]
        )
        self._window_means = np.array(
            [
                state.window_mean if (state is not None and not state.is_in_learning_period) else np.nan
                for state in self._states
            ]
        )
        self._window_sizes = np.array(
            [
                state.window_size if (state is not None and not state.is_in_learning_period) else 1
                for state in self._states
            ]
        )

        # Calculate control limits: mu +/- k * sigma / sqrt(w)
        band_scale = self._band_opts["band_size"]
        self._half_band = band_scale * self._stds / np.sqrt(self._window_sizes)
        self._upper_limits = self._means + self._half_band
        self._lower_limits = self._means - self._half_band

        return self

    def set_plot_opts(self, **options: Unpack[ShewhartStatePlotOpts]) -> Self:
        """
        Set general plot options for Shewhart state subplot.

        Parameters
        ----------
        **options
            xlabel
                X-axis label.
            ylabel
                Y-axis label.
            grid
                Whether to show grid lines.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._plot_opts.update(options)
        return self

    def set_draw_opts(self, **options: Unpack[ShewhartStateDrawOpts]) -> Self:
        """
        Set drawing options for Shewhart state lines.

        Parameters
        ----------
        **options
            mean_color
                Color of the running mean line.
            mean_linewidth
                Width of the running mean line.
            mean_label
                Legend label for running mean.
            control_limit_color
                Color of the control limit lines.
            control_limit_linestyle
                Line style for control limits.
            control_limit_linewidth
                Width of the control limit lines.
            control_limit_label
                Legend label for control limits.
            window_mean_color
                Color of the window mean line.
            window_mean_linewidth
                Width of the window mean line.
            window_mean_label
                Legend label for window mean.
            fill_alpha
                Opacity of the fill between control limits.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._draw_opts.update(options)
        return self

    def set_band_opts(self, **options: Unpack[ShewhartStateBandOpts]) -> Self:
        """
        Set band calculation options for control limits.

        Parameters
        ----------
        **options
            band_size
                Multiplier k for control limits (mu +/- k * sigma / sqrt(w)).
                Default is 3.0.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._band_opts.update(options)
        # Recalculate control limits with new band size
        if hasattr(self, "_states") and self._states:
            band_scale = self._band_opts["band_size"]
            self._half_band = band_scale * self._stds / np.sqrt(self._window_sizes)
            self._upper_limits = self._means + self._half_band
            self._lower_limits = self._means - self._half_band
        return self

    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxMapping) -> PltFigure:
        """Draw using Matplotlib backend."""
        if "shewhart_state" not in axes:
            return figure

        if len(self._states) == 0:
            return figure

        ax = axes["shewhart_state"]

        # Draw window mean first (bottom layer)
        ax.plot(
            self._time_points,
            self._window_means,
            color=self._draw_opts["window_mean_color"],
            linewidth=self._draw_opts["window_mean_linewidth"],
            label=self._draw_opts["window_mean_label"],
            zorder=1,
        )

        # Draw fill between control limits
        ax.fill_between(
            self._time_points,
            self._lower_limits,
            self._upper_limits,
            alpha=self._draw_opts["fill_alpha"],
            color=self._draw_opts["control_limit_color"],
            label="Control Band",
            zorder=2,
        )

        # Draw control limits (upper and lower)
        ax.plot(
            self._time_points,
            self._upper_limits,
            color=self._draw_opts["control_limit_color"],
            linestyle=translate_linestyle(self._draw_opts["control_limit_linestyle"]),
            linewidth=self._draw_opts["control_limit_linewidth"],
            label=self._draw_opts["control_limit_label"],
            zorder=3,
        )

        ax.plot(
            self._time_points,
            self._lower_limits,
            color=self._draw_opts["control_limit_color"],
            linestyle=translate_linestyle(self._draw_opts["control_limit_linestyle"]),
            linewidth=self._draw_opts["control_limit_linewidth"],
            zorder=3,
        )

        # Draw running mean on top
        ax.plot(
            self._time_points,
            self._means,
            color=self._draw_opts["mean_color"],
            linewidth=self._draw_opts["mean_linewidth"],
            label=self._draw_opts["mean_label"],
            zorder=4,
        )

        # Configure axes
        ax.set_xlabel(self._plot_opts["xlabel"])
        ax.set_ylabel(self._plot_opts["ylabel"])

        if self._plot_opts["grid"]:
            ax.grid(True, alpha=MPL_GRID_ALPHA, zorder=0)

        # Set title if provided
        if self._plot_opts["title"] is not None:
            ax.set_title(self._plot_opts["title"])

        ax.legend(loc="best")

        return figure

    def _draw_plotly(self, figure: GoFigure, axes: GoAxMapping) -> GoFigure:
        """Draw using Plotly backend."""
        if "shewhart_state" not in axes:
            return figure
        if len(self._states) == 0:
            return figure

        row, col = axes["shewhart_state"]

        # Draw window mean first (bottom layer)
        figure.add_trace(
            go.Scatter(
                x=self._time_points,
                y=self._window_means,
                mode="lines",
                name=self._draw_opts["window_mean_label"],
                line={
                    "color": self._draw_opts["window_mean_color"],
                    "width": self._draw_opts["window_mean_linewidth"],
                },
            ),
            row=row,
            col=col,
        )

        # Draw fill between control limits
        # figure.add_trace(
        #     go.Scatter(
        #         x=np.concatenate([self._time_points, self._time_points[::-1]]),
        #         y=np.concatenate([self._upper_limits, self._lower_limits[::-1]]),
        #         fill="toself",
        #         fillcolor=f"rgba(255, 0, 0, {self._draw_opts['fill_alpha']})",
        #         line={"color": "rgba(255, 0, 0, 0)"},
        #         name="Control Band",
        #         showlegend=True,
        #     ),
        #     row=row,
        #     col=col,
        # )

        # Draw upper control limit
        figure.add_trace(
            go.Scatter(
                x=self._time_points,
                y=self._lower_limits,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(0, 0, 0, 0)",
                name=self._draw_opts["control_limit_label"],
                line={
                    "color": self._draw_opts["control_limit_color"],
                    "dash": self._draw_opts["control_limit_linestyle"],
                    "width": self._draw_opts["control_limit_linewidth"],
                },
            ),
            row=row,
            col=col,
        )

        # Draw lower control limit (without label to avoid duplicate legend entries)
        figure.add_trace(
            go.Scatter(
                x=self._time_points,
                y=self._upper_limits,
                mode="lines",
                fill="tonexty",
                fillcolor=f"rgba(255, 0, 0, {self._draw_opts['fill_alpha']})",
                name="Control Band",
                showlegend=True,
                line={
                    "color": self._draw_opts["control_limit_color"],
                    "dash": self._draw_opts["control_limit_linestyle"],
                    "width": self._draw_opts["control_limit_linewidth"],
                },
            ),
            row=row,
            col=col,
        )

        # Draw running mean on top
        figure.add_trace(
            go.Scatter(
                x=self._time_points,
                y=self._means,
                mode="lines",
                name=self._draw_opts["mean_label"],
                line={
                    "color": self._draw_opts["mean_color"],
                    "width": self._draw_opts["mean_linewidth"],
                },
            ),
            row=row,
            col=col,
        )

        # Configure axes
        figure.update_xaxes(
            title_text=self._plot_opts["xlabel"],
            showgrid=self._plot_opts["grid"],
            gridwidth=PLOTLY_GRID_WIDTH,
            gridcolor=PLOTLY_GRID_COLOR,
            row=row,
            col=col,
        )
        figure.update_yaxes(
            title_text=self._plot_opts["ylabel"],
            showgrid=self._plot_opts["grid"],
            gridwidth=PLOTLY_GRID_WIDTH,
            gridcolor=PLOTLY_GRID_COLOR,
            row=row,
            col=col,
        )

        # Set title if provided
        if self._plot_opts["title"] is not None:
            annotation_index = get_plotly_subplot_annotation_index(figure, row, col)
            if len(figure.layout.annotations) > annotation_index:
                figure.layout.annotations[annotation_index].text = self._plot_opts["title"]

        return figure
