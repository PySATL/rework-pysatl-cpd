# -*- coding: ascii -*-
"""
Online trace visualizer implementation.

This module provides concrete visualizer for rendering online detection
traces including detection function values and processing times.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Self, Unpack

import numpy as np
import plotly.graph_objects as go

from pysatl_cpd.analysis.visualization.abstracts import ITraceVisualizer
from pysatl_cpd.analysis.visualization.online.states import IOnlineStateVisualizer
from pysatl_cpd.analysis.visualization.specs import FilledLineSpec, LineSpec, PlotSpec
from pysatl_cpd.analysis.visualization.typedefs import (
    DrawBackend,
    GoAxes,
    GoAxMapping,
    GoFigure,
    PltAxes,
    PltAxMapping,
    PltFigure,
)
from pysatl_cpd.analysis.visualization.utils import (
    apply_matplotlib_plot_spec,
    apply_plotly_plot_spec,
    filled_line_spec_to_mpl_fill_kwargs,
    filled_line_spec_to_mpl_line_kwargs,
    filled_line_spec_to_plotly_trace_kwargs,
    get_matplotlib_legend_label,
    get_plotly_legend_kwargs,
    line_spec_to_mpl_kwargs,
    line_spec_to_plotly_trace_kwargs,
)
from pysatl_cpd.core.online import OnlineAlgorithmState, OnlineDetectionTrace

# Plotly layout constants
PLOTLY_GRID_WIDTH = 1
PLOTLY_GRID_COLOR = "lightgray"

# Matplotlib constants
MPL_GRID_ALPHA = 0.3


class OnlineTraceVisualizer[StateT: OnlineAlgorithmState](ITraceVisualizer[OnlineDetectionTrace[StateT]]):
    """
    Visualizer for online detection trace results.

    This visualizer renders detection function values and processing times.
    Annotations such as change points, learning periods, and skip periods
    should be added by the caller using separate visual components.

    Parameters
    ----------
    backend
        Plotting backend to use for rendering.
    state_visualizer
        Visualizer for algorithm state evolution.
    """

    def __init__(
        self,
        backend: DrawBackend,
        state_visualizer: IOnlineStateVisualizer[StateT],
    ) -> None:
        super().__init__(backend)
        self._trace: OnlineDetectionTrace[StateT] | None = None
        self.state_visualizer = state_visualizer

        # Options storage with defaults
        self._detection_func_plot_opts: PlotSpec = {
            "xlabel": "Time Index",
            "ylabel": "Detection Statistic",
            "grid": True,
            "title": None,
        }
        self._detection_func_draw_opts: LineSpec = {
            "color": "blue",
            "linewidth": 1,
            "label": "Detection Function",
            "legend": True,
        }
        self._threshold_draw_opts: LineSpec = {
            "color": "green",
            "linestyle": "solid",
            "linewidth": 1,
            "alpha": 0.5,
            "label": "Threshold",
            "legend": True,
        }
        self._processing_time_plot_opts: PlotSpec = {
            "xlabel": "Time Index",
            "ylabel": "Time (seconds)",
            "grid": True,
            "title": None,
        }
        self._processing_time_draw_opts: FilledLineSpec = {
            "color": "green",
            "linewidth": 1,
            "fill_alpha": 0.3,
            "label": "Processing Time",
            "legend": True,
        }

    def set_trace(self, trace: OnlineDetectionTrace[StateT]) -> Self:
        """
        Set the detection trace to visualize.

        Parameters
        ----------
        trace
            Detection results containing detection function values,
            processing times, and algorithm states.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._trace = trace
        self.state_visualizer.set_states(trace.algorithm_states)
        return self

    @property
    def backend(self) -> DrawBackend:
        """Current drawing backend.

        Returns
        -------
        DrawBackend
        """
        return super().backend

    @backend.setter
    def backend(self, value: str) -> None:
        """Set the drawing backend.

        Propagates the change to the state visualizer.

        Parameters
        ----------
        value
            Backend name.
        """
        from pysatl_cpd.analysis.visualization.abstracts.ivisualizer import IVisualizer

        IVisualizer.backend.fset(self, value)  # type: ignore[attr-defined]
        self.state_visualizer.backend = value

    def set_state_visualizer(self, state_visualizer: IOnlineStateVisualizer[StateT]) -> Self:
        """Replace the state visualizer used by this trace visualizer.

        Parameters
        ----------
        state_visualizer
            New state visualizer instance.

        Returns
        -------
        Self
        """
        state_visualizer.backend = self.backend
        self.state_visualizer = state_visualizer
        if self._trace is not None:
            self.state_visualizer.set_states(self._trace.algorithm_states)
        return self

    def set_detection_func_plot_opts(self, **options: Unpack[PlotSpec]) -> Self:
        """
        Set general plot options for detection function subplot.

        Parameters
        ----------
        **options
            xlabel : str
                X-axis label.
            ylabel : str
                Y-axis label.
            grid : bool
                Whether to show grid lines.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._detection_func_plot_opts.update(options)
        return self

    def set_detection_func_draw_opts(self, **options: Unpack[LineSpec]) -> Self:
        """
        Set drawing options for detection function line.

        Parameters
        ----------
        **options
            color : str
                Line color.
            linewidth : float
                Line width in points.
            label : str
                Legend label for the detection function line.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._detection_func_draw_opts.update(options)
        return self

    def set_threshold_draw_opts(self, **options: Unpack[LineSpec]) -> Self:
        """
        Set drawing options for threshold line.

        Parameters
        ----------
        **options
            color : str
                Line color.
            linestyle : str
                Line style ('solid', 'dash', etc.).
            linewidth : float
                Line width in points.
            alpha : float
                Line opacity between 0 and 1.
            label : str
                Legend label for the threshold line.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._threshold_draw_opts.update(options)
        return self

    def set_processing_time_plot_opts(self, **options: Unpack[PlotSpec]) -> Self:
        """
        Set general plot options for processing time subplot.

        Parameters
        ----------
        **options
            xlabel : str
                X-axis label.
            ylabel : str
                Y-axis label.
            grid : bool
                Whether to show grid lines.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._processing_time_plot_opts.update(options)
        return self

    def set_processing_time_draw_opts(self, **options: Unpack[FilledLineSpec]) -> Self:
        """
        Set drawing options for processing time line.

        Parameters
        ----------
        **options
            color : str
                Line color.
            linewidth : float
                Line width in points.
            fill_alpha : float
                Opacity of fill under the line between 0 and 1.
            label : str
                Legend label for the processing time line.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._processing_time_draw_opts.update(options)
        return self

    @property
    def axes(self) -> set[str]:
        """
        Declare the subplot names required by this visualizer.

        Returns
        -------
        set[str]
            Set containing "detection_function", "processing_time" subplot names,
            and axes from state visualizer.
        """
        return {"detection_function", "processing_time"} | self.state_visualizer.axes

    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxMapping) -> PltFigure:
        """Draw using Matplotlib backend."""
        if self._trace is None:
            return figure

        # Draw detection function
        if "detection_function" in axes:
            self._mpl_draw_detection_function(axes["detection_function"])

        # Draw processing time
        if "processing_time" in axes:
            self._mpl_draw_processing_time(axes["processing_time"])

        # Draw state evolution
        state_axes = {k: v for k, v in axes.items() if k in self.state_visualizer.axes}
        if state_axes:
            self.state_visualizer._draw_matplotlib(figure, state_axes)

        return figure

    def _mpl_draw_detection_function(self, ax: PltAxes) -> None:
        """
        Draw detection function on Matplotlib axes.

        Parameters
        ----------
        ax
            The Matplotlib axes to draw on.
        """
        if self._trace is None:
            return

        time_points = list(range(len(self._trace.detection_function)))

        # Draw detection function line
        ax.plot(
            time_points,
            self._trace.detection_function,
            label=get_matplotlib_legend_label(self._detection_func_draw_opts),
            **line_spec_to_mpl_kwargs(self._detection_func_draw_opts),
        )

        # Draw threshold line
        if self._trace.threshold is not None and not np.isnan(self._trace.threshold):
            ax.axhline(
                y=self._trace.threshold,
                label=get_matplotlib_legend_label(self._threshold_draw_opts),
                **line_spec_to_mpl_kwargs(self._threshold_draw_opts),
            )

        apply_matplotlib_plot_spec(ax, self._detection_func_plot_opts, grid_alpha=MPL_GRID_ALPHA)

        _, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="best")

    def _mpl_draw_processing_time(self, ax: PltAxes) -> None:
        """
        Draw processing time on Matplotlib axes.

        Parameters
        ----------
        ax
            The Matplotlib axes to draw on.
        """
        if self._trace is None:
            return

        time_points = list(range(len(self._trace.processing_time)))

        ax.plot(
            time_points,
            self._trace.processing_time,
            label=get_matplotlib_legend_label(self._processing_time_draw_opts),
            **filled_line_spec_to_mpl_line_kwargs(self._processing_time_draw_opts),
        )
        ax.fill_between(
            time_points,
            0,
            self._trace.processing_time,
            **filled_line_spec_to_mpl_fill_kwargs(self._processing_time_draw_opts),
        )

        apply_matplotlib_plot_spec(ax, self._processing_time_plot_opts, grid_alpha=MPL_GRID_ALPHA)

        _, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="best")

    def _draw_plotly(self, figure: GoFigure, axes: GoAxMapping) -> GoFigure:
        """Draw using Plotly backend."""
        if self._trace is None:
            return figure

        # Draw detection function
        if "detection_function" in axes:
            self._plotly_draw_detection_function(figure, axes["detection_function"])

        # Draw processing time
        if "processing_time" in axes:
            self._plotly_draw_processing_time(figure, axes["processing_time"])

        # Draw state evolution
        state_axes = {k: v for k, v in axes.items() if k in self.state_visualizer.axes}
        if state_axes:
            self.state_visualizer._draw_plotly(figure, state_axes)

        return figure

    def _plotly_draw_detection_function(
        self,
        figure: GoFigure,
        ax_pos: GoAxes,
    ) -> None:
        """
        Draw detection function on Plotly subplot.

        Parameters
        ----------
        figure
            The Plotly figure containing the subplot.
        ax_pos
            The subplot position (row, column) to draw on.
        """
        if self._trace is None:
            return

        row, col = ax_pos
        time_points = list(range(len(self._trace.detection_function)))

        # Draw detection function line
        figure.add_trace(
            go.Scatter(
                x=time_points,
                y=self._trace.detection_function,
                mode="lines",
                **line_spec_to_plotly_trace_kwargs(self._detection_func_draw_opts),
                **get_plotly_legend_kwargs(self._detection_func_draw_opts),
                hovertemplate="Step: %{x}, Value: %{y}",
            ),
            row=row,
            col=col,
        )

        # Draw threshold line
        if self._trace.threshold is not None and not np.isnan(self._trace.threshold):
            figure.add_trace(
                go.Scatter(
                    x=[time_points[0], time_points[-1]] if time_points else [0, 0],
                    y=[self._trace.threshold, self._trace.threshold],
                    mode="lines",
                    **line_spec_to_plotly_trace_kwargs(self._threshold_draw_opts),
                    **get_plotly_legend_kwargs(self._threshold_draw_opts),
                ),
                row=row,
                col=col,
            )

        apply_plotly_plot_spec(
            figure,
            row,
            col,
            self._detection_func_plot_opts,
            grid_width=PLOTLY_GRID_WIDTH,
            grid_color=PLOTLY_GRID_COLOR,
        )

    def _plotly_draw_processing_time(
        self,
        figure: GoFigure,
        ax_pos: GoAxes,
    ) -> None:
        """
        Draw processing time on Plotly subplot.

        Parameters
        ----------
        figure
            The Plotly figure containing the subplot.
        ax_pos
            The subplot position (row, column) to draw on.
        """
        if self._trace is None:
            return

        row, col = ax_pos
        time_points = list(range(len(self._trace.processing_time)))

        figure.add_trace(
            go.Scatter(
                x=time_points,
                y=self._trace.processing_time,
                mode="lines",
                **filled_line_spec_to_plotly_trace_kwargs(self._processing_time_draw_opts),
                **get_plotly_legend_kwargs(self._processing_time_draw_opts),
            ),
            row=row,
            col=col,
        )

        apply_plotly_plot_spec(
            figure,
            row,
            col,
            self._processing_time_plot_opts,
            grid_width=PLOTLY_GRID_WIDTH,
            grid_color=PLOTLY_GRID_COLOR,
        )
