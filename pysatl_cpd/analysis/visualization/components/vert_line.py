# -*- coding: ascii -*-
"""
Vertical line visualizers component.

This module provides a component for visualizing vertical lines at specified
x-coordinates with configurable styling.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Sequence
from typing import Self, Unpack

import plotly.graph_objects as go

from pysatl_cpd.analysis.visualization.abstracts import IVisualComponent
from pysatl_cpd.analysis.visualization.specs import LineSpec
from pysatl_cpd.analysis.visualization.typedefs import DrawBackend, GoAxes, GoFigure, PltAxes, PltFigure
from pysatl_cpd.analysis.visualization.utils import (
    get_matplotlib_legend_label,
    get_plotly_legend_kwargs,
    get_subplot_y_limits,
    line_spec_to_mpl_kwargs,
    line_spec_to_plotly_trace_kwargs,
)


class VerticalLineVisualComponent(IVisualComponent):
    """
    Component for visualizing vertical lines at specified coordinates.

    This component draws vertical lines at given x-coordinates. It supports
    both Matplotlib and Plotly backends with configurable line styles,
    colors, and transparency.

    Parameters
    ----------
    backend
        The plotting backend to use for rendering.
    """

    def __init__(self, backend: DrawBackend) -> None:
        super().__init__(backend)
        self._lines: list[float] = []

        self._style: LineSpec = {
            "color": "blue",
            "linestyle": "dash",
            "linewidth": 2,
            "alpha": 0.8,
            "legend": True,
        }

    def set_lines(self, indices: Sequence[float]) -> Self:
        """
        Set vertical line positions.

        Parameters
        ----------
        indices
            X-axis coordinates where vertical lines should be drawn.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._lines = list(indices)
        return self

    def set_style(self, **style: Unpack[LineSpec]) -> Self:
        """
        Set style for vertical lines.

        Parameters
        ----------
        **style
            color : str
                Line color. Accepts any matplotlib color name, hex code, or RGB tuple.
            linestyle : str
                For Matplotlib: line style ('--', '-.', '-', ':').
                For Plotly: line style ('solid', 'dot', 'dash', 'longdash',
                'dashdot', 'longdashdot').
            linewidth : float
                Line width in points.
            alpha : float
                Line opacity between 0 (transparent) and 1 (opaque).

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._style.update(style)
        return self

    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxes, add_legend: bool = False) -> None:
        """
        Draw vertical lines on Matplotlib axes.

        Parameters
        ----------
        figure
            The Matplotlib figure containing the axes.
        axes
            The Matplotlib axes to draw on.
        add_legend
            Whether to add legend entry for these lines.
        """
        if not self._lines:
            return

        plot_opts = line_spec_to_mpl_kwargs(self._style)

        for i, idx in enumerate(self._lines):
            axes.axvline(
                idx,
                **plot_opts,
                label=get_matplotlib_legend_label(self._style, allow_legend=add_legend) if i == 0 else None,
            )

    def _draw_plotly(self, figure: GoFigure, axes: GoAxes, add_legend: bool = False) -> None:
        """
        Draw vertical lines on Plotly subplot.

        Parameters
        ----------
        figure
            The Plotly figure containing the subplot.
        axes
            The subplot position (row, column) to draw on.
        add_legend
            Whether to add legend entry for these lines.
        """
        if not self._lines:
            return

        # Get y-axis limits
        y_min, y_max = get_subplot_y_limits(figure, axes)

        spacing = 11
        y_hover_poses = [y_min + float(x) / spacing * (y_max - y_min) for x in range(spacing)]

        # Build coordinates for all vertical lines
        x_coords = []
        y_coords = []
        hover_x = []
        hover_y = []
        hover_t = []
        for idx in self._lines:
            x_coords.extend([idx, idx, None])
            y_coords.extend([y_min, y_max, None])

            # Add multiple hover points along the line (every 10% of height)
            hover_x.extend([idx] * spacing)
            hover_y.extend(y_hover_poses)
            hover_label = self._style.get("label", "Vertical Line")
            hover_t.extend([f"x={idx:.0f}<br>{hover_label}"] * spacing)

        # trace (no hover)
        figure.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                **line_spec_to_plotly_trace_kwargs(self._style),
                **get_plotly_legend_kwargs(self._style, allow_legend=add_legend),
                hoverinfo="skip",  # Line doesn't show hover
            ),
            row=axes[0],
            col=axes[1],
        )

        # Add invisible points for hover detection
        figure.add_trace(
            go.Scatter(
                x=hover_x,
                y=hover_y,
                mode="markers",
                marker={
                    "color": self._style["color"],
                    "size": 8,
                    "opacity": 0,  # Completely invisible
                },
                hoverinfo="text",
                hovertext=hover_t,
                hovertemplate="%{hovertext}<extra></extra>",
                **get_plotly_legend_kwargs(self._style, allow_legend=add_legend, primary_trace=False),
            ),
            row=axes[0],
            col=axes[1],
        )

    def clear(self) -> Self:
        """
        Clear all vertical lines.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._lines.clear()
        return self
