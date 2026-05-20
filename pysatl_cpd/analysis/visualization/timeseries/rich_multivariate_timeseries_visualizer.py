# -*- coding: ascii -*-
"""Rich pandas-first multivariate time-series visualizer."""

from __future__ import annotations

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections import defaultdict
from typing import Any, NamedTuple, Self, Unpack

import pandas as pd
import plotly.graph_objs as go

from pysatl_cpd.analysis.visualization.specs import LineSpec, PlotSpec
from pysatl_cpd.analysis.visualization.timeseries.abstract_multivariate_timeseries_visualizer import (
    AbstractMultivariateTimeseriesVisualizer,
)
from pysatl_cpd.analysis.visualization.timeseries.univariate_timeseries_visualizer import (
    MPL_GRID_ALPHA,
    PLOTLY_GRID_COLOR,
    PLOTLY_GRID_WIDTH,
)
from pysatl_cpd.analysis.visualization.typedefs import DrawBackend, GoAxMapping, GoFigure, PltAxMapping, PltFigure
from pysatl_cpd.analysis.visualization.utils import (
    apply_matplotlib_plot_spec,
    apply_matplotlib_twin_plot_spec,
    apply_plotly_plot_spec,
    apply_plotly_twin_plot_spec,
    get_matplotlib_legend_label,
    get_plotly_legend_kwargs,
    line_spec_to_mpl_kwargs,
    line_spec_to_plotly_trace_kwargs,
)
from pysatl_cpd.data import DataProvider
from pysatl_cpd.data.typedefs import TimeseriesAnnotation


class _BoundSeries(NamedTuple):
    series_name: str
    column: str
    twin: bool


class RichMultivariateTimeseriesVisualizer(
    AbstractMultivariateTimeseriesVisualizer[DataProvider[Any, TimeseriesAnnotation]]
):
    """Render pandas-backed multivariate data as logical plots with logical series."""

    def __init__(self, backend: DrawBackend | str) -> None:
        super().__init__(backend)
        self._plot_specs: dict[str, PlotSpec] = {}
        self._series_specs: dict[tuple[str, str], LineSpec] = {}
        self._plot_series: dict[str, list[_BoundSeries]] = defaultdict(list)

    @property
    def axes(self) -> set[str]:
        """Return logical plot names required by this visualizer."""
        return set(self._plot_specs) | set(self._plot_series)

    @property
    def ordered_axes(self) -> list[str]:
        """Return logical plot names in insertion order."""
        ordered_axes = list(self._plot_specs)
        for plot_name in self._plot_series:
            if plot_name not in ordered_axes:
                ordered_axes.append(plot_name)
        return ordered_axes

    @property
    def axes_with_twin(self) -> set[str]:
        """Return logical plots that contain at least one twin-axis series."""
        return {
            plot_name
            for plot_name, bound_series in self._plot_series.items()
            if any(series.twin for series in bound_series)
        }

    def define_plot(self, plot_name: str, **plot_spec_kwargs: Unpack[PlotSpec]) -> Self:
        """Define or update a logical plot and its visual spec."""
        self._ensure_plot(plot_name)
        self._plot_specs[plot_name].update(plot_spec_kwargs)
        return self

    def add_series(
        self,
        plot_name: str,
        series_name: str,
        *,
        column: str,
        twin: bool = False,
        **line_spec_kwargs: Unpack[LineSpec],
    ) -> Self:
        """Bind one provider column to one logical plot series."""
        self._ensure_plot(plot_name)
        self._validate_series_binding(column)

        bound_series = _BoundSeries(series_name=series_name, column=column, twin=twin)
        existing = self._plot_series[plot_name]
        existing = [series for series in existing if series.series_name != series_name]
        existing.append(bound_series)
        self._plot_series[plot_name] = existing

        series_key = (plot_name, series_name)
        spec = self._series_specs.get(series_key, self._make_default_line_opts(series_name))
        spec.update(line_spec_kwargs)
        self._series_specs[series_key] = spec
        return self

    def set_plot_opts(self, plot_name: str, **plot_spec_kwargs: Unpack[PlotSpec]) -> Self:
        """Set plot visuals for one logical plot."""
        self._ensure_plot(plot_name)
        self._plot_specs[plot_name].update(plot_spec_kwargs)
        return self

    def set_series_opts(self, plot_name: str, series_name: str, **line_spec_kwargs: Unpack[LineSpec]) -> Self:
        """Set visual options for one logical series."""
        series_key = (plot_name, series_name)
        if series_key not in self._series_specs:
            raise ValueError(f"Unknown series '{series_name}' in plot '{plot_name}'")
        self._series_specs[series_key].update(line_spec_kwargs)
        return self

    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxMapping) -> PltFigure:
        """Draw configured logical plots using Matplotlib."""
        dataset = self._require_dataset()
        data_provider = self._require_data_provider()
        time_points = self._resolve_time_points(data_provider, len(dataset))

        for plot_name in self.axes:
            if plot_name not in axes:
                continue

            ax = axes[plot_name]
            plot_spec = self._plot_specs.get(plot_name, self._make_default_plot_opts())
            twin_ax = None

            for bound_series in self._plot_series.get(plot_name, []):
                target_ax = ax
                if bound_series.twin:
                    if twin_ax is None:
                        twin_ax = ax.twinx()
                    target_ax = twin_ax

                series_key = (plot_name, bound_series.series_name)
                series_spec = self._series_specs[series_key]
                target_ax.plot(
                    time_points,
                    dataset[bound_series.column].tolist(),
                    label=get_matplotlib_legend_label(series_spec),
                    **line_spec_to_mpl_kwargs(series_spec),
                )

            apply_matplotlib_plot_spec(ax, plot_spec, grid_alpha=MPL_GRID_ALPHA)
            if twin_ax is not None:
                apply_matplotlib_twin_plot_spec(twin_ax, plot_spec)
                handles, labels = ax.get_legend_handles_labels()
                twin_handles, twin_labels = twin_ax.get_legend_handles_labels()
                handles.extend(twin_handles)
                labels.extend(twin_labels)
                if labels:
                    ax.legend(handles, labels, loc="best")

        return figure

    def _draw_plotly(self, figure: GoFigure, axes: GoAxMapping) -> GoFigure:
        """Draw configured logical plots using Plotly."""
        dataset = self._require_dataset()
        data_provider = self._require_data_provider()
        time_points = self._resolve_time_points(data_provider, len(dataset))

        for plot_name in self.axes:
            if plot_name not in axes:
                continue

            row, col = axes[plot_name]
            plot_spec = self._plot_specs.get(plot_name, self._make_default_plot_opts())
            has_twin = False

            for bound_series in self._plot_series.get(plot_name, []):
                series_key = (plot_name, bound_series.series_name)
                series_spec = self._series_specs[series_key]
                has_twin = has_twin or bound_series.twin
                figure.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=dataset[bound_series.column].tolist(),
                        mode="lines",
                        **line_spec_to_plotly_trace_kwargs(series_spec),
                        **get_plotly_legend_kwargs(series_spec),
                    ),
                    row=row,
                    col=col,
                    secondary_y=bound_series.twin,
                )

            apply_plotly_plot_spec(
                figure,
                row,
                col,
                plot_spec,
                grid_width=PLOTLY_GRID_WIDTH,
                grid_color=PLOTLY_GRID_COLOR,
            )
            if has_twin:
                apply_plotly_twin_plot_spec(figure, row, col, plot_spec)

        return figure

    def _validate_provider(self, data_provider: DataProvider[Any, TimeseriesAnnotation]) -> None:
        """Ensure the provider exposes a pandas dataset."""
        if self._get_provider_dataset(data_provider) is None:
            raise TypeError("RichMultivariateTimeseriesVisualizer requires a pandas-backed provider")

    def _validate_time_column(
        self,
        data_provider: DataProvider[Any, TimeseriesAnnotation] | None,
        time_column: str | None,
    ) -> None:
        """Ensure the configured time column exists on pandas-backed providers."""
        if time_column is None or data_provider is None:
            return

        dataset = self._get_provider_dataset(data_provider)
        if dataset is None:
            raise TypeError("RichMultivariateTimeseriesVisualizer requires a pandas-backed provider")
        if time_column not in dataset.columns:
            raise ValueError(f"Unknown time column '{time_column}'. Available columns: {list(dataset.columns)}")

    def _ensure_plot(self, plot_name: str) -> None:
        """Create a default plot spec if the logical plot is new."""
        if plot_name not in self._plot_specs:
            self._plot_specs[plot_name] = self._make_default_plot_opts()

    def _validate_series_binding(self, column: str) -> None:
        """Ensure a bound series references a valid dataframe column."""
        if self._data_provider is None:
            return

        dataset = self._require_dataset()
        if column not in dataset.columns:
            raise ValueError(f"Unknown series column '{column}'. Available columns: {list(dataset.columns)}")

    def _require_data_provider(self) -> DataProvider[Any, TimeseriesAnnotation]:
        """Return the stored provider or fail with a clear error."""
        if self._data_provider is None:
            raise ValueError("A pandas-backed data provider must be set before drawing")
        return self._data_provider

    def _require_dataset(self) -> pd.DataFrame:
        """Return the provider dataset or fail with a clear error."""
        dataset = self._get_provider_dataset(self._require_data_provider())
        if dataset is None:
            raise TypeError("RichMultivariateTimeseriesVisualizer requires a pandas-backed provider")
        return dataset
