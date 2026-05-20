# -*- coding: ascii -*-
"""Plain multivariate time-series visualizer."""

from __future__ import annotations

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Sequence
from typing import Any, Self, Unpack, cast

import numpy as np
import plotly.graph_objs as go

from pysatl_cpd.analysis.visualization.specs import LineSpec, PlotSpec
from pysatl_cpd.analysis.visualization.timeseries.abstract_multivariate_timeseries_visualizer import (
    AbstractMultivariateTimeseriesVisualizer,
    AxisSelector,
)
from pysatl_cpd.analysis.visualization.timeseries.univariate_timeseries_visualizer import (
    MPL_GRID_ALPHA,
    PLOTLY_GRID_COLOR,
    PLOTLY_GRID_WIDTH,
)
from pysatl_cpd.analysis.visualization.typedefs import DrawBackend, GoAxMapping, GoFigure, PltAxMapping, PltFigure
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
from pysatl_cpd.typedefs import MultivariateNumericArray, UnivariateNumericArray


class PlainMultivariateTimeseriesVisualizer(
    AbstractMultivariateTimeseriesVisualizer[DataProvider[UnivariateNumericArray, TimeseriesAnnotation]]
):
    """Render one multivariate dimension per subplot."""

    def __init__(self, backend: DrawBackend | str, dimensionality: int) -> None:
        super().__init__(backend, dimensionality=dimensionality)
        self._plot_opts: list[PlotSpec] = [self._make_default_plot_opts() for _ in range(dimensionality)]
        self._draw_opts: list[LineSpec] = [
            self._make_default_line_opts(f"Dimension {index}", color="black") for index in range(dimensionality)
        ]

    @property
    def axes(self) -> set[str]:
        """Return one required subplot name per dimension."""
        return {self._axis_name(index) for index in range(cast(int, self._dimensionality))}

    @property
    def ordered_axes(self) -> list[str]:
        """Return timeseries axes in deterministic dimension order."""
        return [self._axis_name(index) for index in range(cast(int, self._dimensionality))]

    def set_plot_opts(self, *, axes: Sequence[AxisSelector], **options: Unpack[PlotSpec]) -> Self:
        """Set plot visuals for selected dimensions."""
        for axis_index in self._resolve_axis_indices(axes):
            self._plot_opts[axis_index].update(options)
        return self

    def set_draw_opts(self, *, axes: Sequence[AxisSelector], **options: Unpack[LineSpec]) -> Self:
        """Set line visuals for selected dimensions."""
        for axis_index in self._resolve_axis_indices(axes):
            self._draw_opts[axis_index].update(options)
        return self

    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxMapping) -> PltFigure:
        """Draw each dimension on its own Matplotlib axes."""
        series = self._get_series_matrix()
        if series is None or self._data_provider is None:
            return figure

        time_points = self._resolve_time_points(self._data_provider, series.shape[0])
        for axis_index in range(cast(int, self._dimensionality)):
            axis_name = self._axis_name(axis_index)
            if axis_name not in axes:
                continue

            ax = axes[axis_name]
            plot_opts = self._plot_opts[axis_index]
            draw_opts = self._draw_opts[axis_index]
            ax.plot(
                time_points,
                series[:, axis_index],
                label=get_matplotlib_legend_label(draw_opts),
                **line_spec_to_mpl_kwargs(draw_opts),
            )
            apply_matplotlib_plot_spec(ax, plot_opts, grid_alpha=MPL_GRID_ALPHA)

        return figure

    def _draw_plotly(self, figure: GoFigure, axes: GoAxMapping) -> GoFigure:
        """Draw each dimension on its own Plotly subplot."""
        series = self._get_series_matrix()
        if series is None or self._data_provider is None:
            return figure

        time_points = self._resolve_time_points(self._data_provider, series.shape[0])
        for axis_index in range(cast(int, self._dimensionality)):
            axis_name = self._axis_name(axis_index)
            if axis_name not in axes:
                continue

            row, col = axes[axis_name]
            plot_opts = self._plot_opts[axis_index]
            draw_opts = self._draw_opts[axis_index]
            figure.add_trace(
                go.Scatter(
                    x=time_points,
                    y=series[:, axis_index],
                    mode="lines",
                    **line_spec_to_plotly_trace_kwargs(draw_opts),
                    **get_plotly_legend_kwargs(draw_opts),
                ),
                row=row,
                col=col,
            )
            apply_plotly_plot_spec(
                figure,
                row,
                col,
                plot_opts,
                grid_width=PLOTLY_GRID_WIDTH,
                grid_color=PLOTLY_GRID_COLOR,
            )

        return figure

    def _validate_provider(self, data_provider: DataProvider[UnivariateNumericArray, TimeseriesAnnotation]) -> None:
        """Validate that provider dimensionality matches this visualizer."""
        self._extract_series_matrix(data_provider)

    def _get_series_matrix(self) -> MultivariateNumericArray | None:
        """Return provider data as a 2-D matrix after validation."""
        if self._data_provider is None:
            return None
        return self._extract_series_matrix(self._data_provider)

    def _extract_series_matrix(
        self,
        data_provider: DataProvider[UnivariateNumericArray, TimeseriesAnnotation],
    ) -> MultivariateNumericArray:
        """Convert provider observations into a validated 2-D matrix."""
        if hasattr(data_provider, "raw_data"):
            raw_data = np.asarray(cast(Any, data_provider).raw_data)
        else:
            rows = list(data_provider)
            raw_data = np.empty((0, cast(int, self._dimensionality)), dtype=float) if not rows else np.asarray(rows)

        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(-1, 1)
        if raw_data.ndim != 2:
            raise ValueError(f"Expected 2-D multivariate data, got array with {raw_data.ndim} dimensions")
        if raw_data.shape[1] != self._dimensionality:
            raise ValueError(
                "Provider dimensionality does not match visualizer dimensionality: "
                f"expected {self._dimensionality}, got {raw_data.shape[1]}"
            )

        return cast(MultivariateNumericArray, raw_data)

    @staticmethod
    def _axis_name(axis_index: int) -> str:
        """Return the subplot name for one dimension."""
        return f"timeseries_{axis_index}"
