# -*- coding: ascii -*-
"""
Univariate time series visualizer implementation.

This module provides a visualizer for rendering univariate time series data
with change point annotations, period fills, and ground truth markers.
"""

from __future__ import annotations

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Self, Unpack

import plotly.graph_objs as go

from pysatl_cpd.analysis.visualization.abstracts import (
    ITimeseriesVisualizer,
)
from pysatl_cpd.analysis.visualization.specs import LineSpec, PlotSpec
from pysatl_cpd.analysis.visualization.typedefs import (
    DrawBackend,
    GoAxMapping,
    GoFigure,
    PltAxMapping,
    PltFigure,
)
from pysatl_cpd.analysis.visualization.utils import (
    apply_matplotlib_plot_spec,
    apply_plotly_plot_spec,
    get_matplotlib_legend_label,
    get_plotly_legend_kwargs,
    line_spec_to_mpl_kwargs,
    line_spec_to_plotly_trace_kwargs,
)
from pysatl_cpd.data import DataProvider
from pysatl_cpd.data.typedefs import TimeseriesAnnotation
from pysatl_cpd.typedefs import Number

# Plotly layout constants
PLOTLY_GRID_COLOR = "lightgray"
PLOTLY_GRID_WIDTH = 1
PLOTLY_TITLE_X = 0.5
PLOTLY_TITLE_Y = 0.95

# Matplotlib constants
MPL_GRID_ALPHA = 0.3


class UnivariateTimeseriesVisualizer(ITimeseriesVisualizer[DataProvider[Number, TimeseriesAnnotation]]):
    """
    Visualizer for univariate time series with change point annotations.

    This visualizer renders the original time series data with optional
    detected change points, forced change points, ground truth markers,
    learning periods, skip periods, and margin windows.

    Parameters
    ----------
    backend
        Plotting backend to use for rendering.
    """

    def __init__(self, backend: DrawBackend) -> None:
        super().__init__(backend)
        self._data_provider: DataProvider[Number, TimeseriesAnnotation] | None = None

        # Options storage with defaults
        self._plot_opts: PlotSpec = {
            "xlabel": "Time Index",
            "ylabel": "Value",
            "grid": True,
            "title": None,
        }
        self._draw_opts: LineSpec = {
            "color": "black",
            "linewidth": 1.5,
            "alpha": 1.0,
            "label": "Time Series",
            "legend": True,
        }

    def set_data_provider(self, data_provider: DataProvider[Number, TimeseriesAnnotation]) -> Self:
        """Set the data provider containing the time series observations.

        Parameters
        ----------
        data_provider
            Provider with univariate numeric observations.

        Returns
        -------
        Self
        """
        self._data_provider = data_provider
        return self

    def set_plot_opts(self, **options: Unpack[PlotSpec]) -> Self:
        """Set general plot options for time series subplot.

        Parameters
        ----------
        **options
            :xlabel: X-axis label.
            :ylabel: Y-axis label.
            :grid: Show grid lines.
            :title: Subplot title.

        Returns
        -------
        Self
        """
        self._plot_opts.update(options)
        return self

    def set_draw_opts(self, **options: Unpack[LineSpec]) -> Self:
        """Set drawing options for time series line.

        Parameters
        ----------
        **options
            :color: Line colour.
            :linewidth: Line width.
            :alpha: Line opacity.
            :label: Legend label.
            :legend: Show legend entry.

        Returns
        -------
        Self
        """
        self._draw_opts.update(options)
        return self

    @property
    def axes(self) -> set[str]:
        """Subplot names required by this visualizer.

        Returns
        -------
        set[str]
        """
        return {"timeseries"}

    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxMapping) -> PltFigure:
        """Draw the time series on a Matplotlib axes.

        Parameters
        ----------
        figure
            Matplotlib figure.
        axes
            Axes mapping containing ``"timeseries"``.

        Returns
        -------
        PltFigure
        """
        if "timeseries" not in axes:
            return figure

        ax = axes["timeseries"]

        # Draw time series data
        if self._data_provider is not None:
            time_points = list(range(len(self._data_provider)))
            values = list(self._data_provider)

            ax.plot(
                time_points,
                values,
                label=get_matplotlib_legend_label(self._draw_opts),
                **line_spec_to_mpl_kwargs(self._draw_opts),
            )

        apply_matplotlib_plot_spec(ax, self._plot_opts, grid_alpha=MPL_GRID_ALPHA)

        return figure

    def _draw_plotly(self, figure: GoFigure, axes: GoAxMapping) -> GoFigure:
        """Draw the time series on a Plotly subplot.

        Parameters
        ----------
        figure
            Plotly figure.
        axes
            Axes mapping containing ``"timeseries"``.

        Returns
        -------
        GoFigure
        """
        if "timeseries" not in axes:
            return figure

        row, col = axes["timeseries"]

        # Draw time series data
        if self._data_provider is not None:
            time_points = list(range(len(self._data_provider)))
            values = list(self._data_provider)

            figure.add_trace(
                go.Scatter(
                    x=time_points,
                    y=values,
                    mode="lines",
                    **line_spec_to_plotly_trace_kwargs(self._draw_opts),
                    **get_plotly_legend_kwargs(self._draw_opts),
                ),
                row=row,
                col=col,
            )

        apply_plotly_plot_spec(
            figure,
            row,
            col,
            self._plot_opts,
            grid_width=PLOTLY_GRID_WIDTH,
            grid_color=PLOTLY_GRID_COLOR,
        )

        return figure
