# -*- coding: ascii -*-
"""
Vertical fill visualizers component.

This module provides a component for visualizing filled vertical regions
between x-coordinates.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Sequence
from typing import Self, Unpack

import plotly.graph_objects as go

from pysatl_cpd.analysis.visualization.abstracts.ivisual_component import IVisualComponent
from pysatl_cpd.analysis.visualization.specs import FillSpec
from pysatl_cpd.analysis.visualization.typedefs import DrawBackend, GoAxes, GoFigure, PltAxes, PltFigure
from pysatl_cpd.analysis.visualization.utils import (
    fill_spec_to_mpl_kwargs,
    fill_spec_to_plotly_trace_kwargs,
    get_matplotlib_legend_label,
    get_plotly_legend_kwargs,
    get_subplot_y_limits,
)


class VerticalFillComponent(IVisualComponent):
    """Component for visualising filled vertical regions between x-coordinates.

    Parameters
    ----------
    backend
        Plotting backend.
    """

    def __init__(self, backend: DrawBackend) -> None:
        super().__init__(backend)
        self._regions: list[tuple[float, float]] = []

        self._style: FillSpec = {
            "fill_color": "gray",
            "fill_alpha": 0.3,
            "legend": True,
        }

    def set_regions(self, regions: Sequence[tuple[float, float]]) -> Self:
        """Set vertical regions to fill.

        Parameters
        ----------
        regions
            Sequence of (start, end) x-coordinate pairs.

        Returns
        -------
        Self
        """
        self._regions = list(regions)
        return self

    def set_style(self, **style: Unpack[FillSpec]) -> Self:
        """Set style for vertical fills.

        Parameters
        ----------
        **style
            :fill_color: Color for the filled region.
            :fill_alpha: Opacity between 0 and 1.

        Returns
        -------
        Self
        """
        self._style.update(style)
        return self

    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxes, add_legend: bool = False) -> None:
        """
        Draw vertical fills on Matplotlib axes.

        Parameters
        ----------
        figure
            The Matplotlib figure containing the axes.
        axes
            The Matplotlib axes to draw on.
        add_legend
            Whether to add legend entry for these fills.
        """
        if not self._regions:
            return
        plot_opts = fill_spec_to_mpl_kwargs(self._style)
        for i, region in enumerate(self._regions):
            axes.axvspan(
                *region,
                **plot_opts,
                label=get_matplotlib_legend_label(self._style, allow_legend=add_legend) if i == 0 else None,
            )

    def _draw_plotly(self, figure: GoFigure, axes: GoAxes, add_legend: bool = False) -> None:
        """
        Draw vertical fills on Plotly subplot.

        Parameters
        ----------
        figure
            The Plotly figure containing the subplot.
        axes
            The subplot position (row, column) to draw on.
        add_legend
            Whether to add legend entry for these fills.
        """
        if not self._regions:
            return
        plot_opts = fill_spec_to_plotly_trace_kwargs(self._style)

        # Build coordinates for all rectangles in one trace
        x_coords = []
        y_coords = []
        y_min, y_max = get_subplot_y_limits(figure, axes)

        # Build hover points
        hover_x = []
        hover_y = []
        hover_t = []
        spacing = 11
        y_hover_poses = [y_min + float(x) / spacing * (y_max - y_min) for x in range(spacing)]

        for start, end in self._regions:
            # Add rectangle coordinates with None separators
            x_coords.extend([start, start, end, end, start, None])
            y_coords.extend([y_min, y_max, y_max, y_min, y_min, None])

            # Add multiple hover points within the rectangle area
            x_mid = (start + end) / 2
            hover_x.extend([x_mid] * spacing)
            hover_y.extend(y_hover_poses)
            hover_label = self._style.get("label", "Vertical Fill")
            hover_t.extend([f"[{start:.0f}, {end:.0f}] {hover_label}"] * spacing)

        # Add rectangle trace (no hover)
        figure.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",  # Rectangle doesn't show hover
                **plot_opts,
                **get_plotly_legend_kwargs(self._style, allow_legend=add_legend),
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
                    "color": self._style["fill_color"],
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
        Clear all vertical regions.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._regions.clear()
        return self
