# -*- coding: ascii -*-
"""
Online Change-Point Detection Plotter.

This module provides the OnlineCpdPlotter class that coordinates multiple
visualizers and components to create comprehensive visualizations for
change-point detection analysis.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Any, Self

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

from pysatl_cpd.analysis.visualization.abstracts import ITimeseriesVisualizer, IVisualComponent
from pysatl_cpd.analysis.visualization.components import VerticalFillComponent, VerticalLineVisualComponent
from pysatl_cpd.analysis.visualization.online import DummyStateVisualizer, OnlineTraceVisualizer
from pysatl_cpd.analysis.visualization.online.plotters.online_cpd_plotter_layouts import (
    DashboardLayout,
    DashboardLiteLayout,
    ILayoutStrategy,
    SplitLayout,
    VerticalLayout,
)
from pysatl_cpd.analysis.visualization.timeseries import UnivariateTimeseriesVisualizer
from pysatl_cpd.analysis.visualization.typedefs import AxMapping, DrawBackend, Figure, GoAxMapping, PltAxMapping
from pysatl_cpd.analysis.visualization.utils import normalize_backend
from pysatl_cpd.core.online import OnlineAlgorithmState, OnlineDetectionTrace
from pysatl_cpd.data import DataProvider
from pysatl_cpd.data.typedefs import TimeseriesAnnotation


class OnlineCpdPlotter[StateT: OnlineAlgorithmState]:
    """
    Coordinator class for online change-point detection visualizations.

    This class manages visualizers and components to create comprehensive
    visualizations. It provides a simplified API for setting up visualizations
    with configurable defaults.

    Parameters
    ----------
    backend
        The plotting backend to use (MATPLOTLIB or PLOTLY).
    data_provider
        The data provider containing observations.
    detection_trace
        The detection trace from online algorithm execution.
    layout
        Layout strategy to use. Can be:

        - An ILayoutStrategy instance
        - One of: "vertical", "split", "dashboard-lite", "dashboard"
        - None (defaults to "vertical")
    """

    def __init__(
        self,
        backend: DrawBackend,
        data_provider: DataProvider[Any, TimeseriesAnnotation] | None = None,
        detection_trace: OnlineDetectionTrace[StateT] | None = None,
        layout: ILayoutStrategy | str | None = None,
    ) -> None:
        self._backend = normalize_backend(backend)
        self._data_provider = data_provider
        self._detection_trace = detection_trace

        # Layout strategy
        self._layout: ILayoutStrategy = self._resolve_layout(layout)

        # Visualizers
        self.timeseries_visualizer: ITimeseriesVisualizer[Any] | None = None
        self.trace_visualizer: OnlineTraceVisualizer[StateT] | None = None

        # Components storage: name -> (component, axes_names, show_legend)
        self._components: dict[str, tuple[IVisualComponent, list[str], bool]] = {}

        # Ground truth configuration
        self._ground_truth_change_points: list[int] = []
        self._ground_truth_margin: tuple[int, int] = (0, 0)

        # Legend display control
        self._legend_axis: str | None = None

        # Initialize defaults
        self._create_default_visualizers()
        self._create_default_components()

    @property
    def backend(self) -> DrawBackend:
        """
        Return the plotting backend used by this visualizer.

        Returns
        -------
        DrawBackend
            Current backend (MATPLOTLIB or PLOTLY).
        """
        return self._backend

    @backend.setter
    def backend(self, value: DrawBackend | str) -> None:
        """Set the plotting backend and propagate to all visualizers and components.

        Parameters
        ----------
        value
            Backend name (``MATPLOTLIB`` or ``PLOTLY``).
        """
        self._backend = normalize_backend(value)
        if self.timeseries_visualizer is not None:
            self.timeseries_visualizer.backend = value
        if self.trace_visualizer is not None:
            self.trace_visualizer.backend = value
        for component, _, _ in self._components.values():
            component.backend = value

    def _resolve_layout(self, layout: ILayoutStrategy | str | None) -> ILayoutStrategy:
        """
        Resolve layout specification to ILayoutStrategy instance.

        Parameters
        ----------
        layout
            Layout specification.

        Returns
        -------
        ILayoutStrategy
            Resolved layout strategy.

        Raises
        ------
        ValueError
            If the layout string is unknown or the layout type is invalid.
        """
        if layout is None:
            return VerticalLayout()
        elif isinstance(layout, ILayoutStrategy):
            return layout
        elif isinstance(layout, str):
            layout = layout.lower()
            if layout == "vertical":
                return VerticalLayout()
            elif layout == "split":
                return SplitLayout()
            elif layout == "dashboard-lite":
                return DashboardLiteLayout()
            elif layout == "dashboard":
                return DashboardLayout()
            else:
                raise ValueError(f"Unknown layout: {layout}")
        else:
            raise ValueError(f"Invalid layout type: {type(layout)}")

    def default_layout(self) -> tuple[Figure, AxMapping]:
        """
        Create figure and axes using the current layout strategy.

        Returns
        -------
        tuple[Figure, AxMapping]
            Figure object and axes mapping for drawing.
        """
        return self._create_figure_and_axes()

    def set_data_provider(self, data_provider: DataProvider[Any, TimeseriesAnnotation]) -> Self:
        """
        Set the data provider for the timeseries visualizer.

        Parameters
        ----------
        data_provider
            Data provider containing observations.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._data_provider = data_provider
        if self.timeseries_visualizer is not None:
            self.timeseries_visualizer.set_data_provider(data_provider)
        return self

    def set_timeseries_visualizer(self, timeseries_visualizer: ITimeseriesVisualizer[Any]) -> Self:
        """Replace the timeseries visualizer coordinated by this plotter."""
        timeseries_visualizer.backend = self.backend
        self.timeseries_visualizer = timeseries_visualizer
        if self._data_provider is not None:
            self.timeseries_visualizer.set_data_provider(self._data_provider)
        return self

    def set_detection_trace(self, detection_trace: OnlineDetectionTrace[StateT]) -> Self:
        """
        Set the detection trace for the trace visualizer.

        Parameters
        ----------
        detection_trace
            Detection trace from online algorithm execution.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._detection_trace = detection_trace
        if self.trace_visualizer is not None:
            self.trace_visualizer.set_trace(detection_trace)
        return self

    def set_ground_truth(self, change_points: list[int], margin: int | tuple[int, int] = 0) -> Self:
        """
        Set ground truth change points and margin.

        Parameters
        ----------
        change_points
            List of ground truth change point indices.
        margin
            Margin around change points. An integer creates a symmetric margin.
            A tuple ``(left, right)`` creates an asymmetric margin. If both
            values are 0, the margin is not drawn.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._ground_truth_change_points = change_points
        normalized_margin = self._normalize_margin(margin)
        self._ground_truth_margin = normalized_margin

        # Update ground_truth_lines component if exists
        if "ground_truth_lines" in self._components:
            component, _, _ = self._components["ground_truth_lines"]
            if hasattr(component, "set_lines"):
                component.set_lines(change_points)

        # Update margin_fill component if margin > 0
        if normalized_margin != (0, 0):
            if "margin_fill" in self._components:
                component, _, _ = self._components["margin_fill"]
                if hasattr(component, "set_regions"):
                    left_margin, right_margin = normalized_margin
                    regions = [(cp - left_margin, cp + right_margin) for cp in change_points]
                    component.set_regions(regions)
        elif "margin_fill" in self._components:
            component, _, _ = self._components["margin_fill"]
            if hasattr(component, "clear"):
                component.clear()

        return self

    def _normalize_margin(self, margin: int | tuple[int, int]) -> tuple[int, int]:
        """Normalise a margin specification to a (left, right) tuple.

        Parameters
        ----------
        margin
            Symmetric margin (int) or asymmetric margin (tuple).

        Returns
        -------
        tuple[int, int]
            Normalised (left, right) margin tuple.

        Raises
        ------
        ValueError
            If margin values are negative or the tuple does not have
            exactly two elements.
        """
        if isinstance(margin, int):
            if margin < 0:
                raise ValueError("Margin must be non-negative")
            return margin, margin

        if len(margin) != 2:
            raise ValueError("Margin tuple must contain exactly two values: (left, right)")

        left_margin, right_margin = margin
        if left_margin < 0 or right_margin < 0:
            raise ValueError("Left and right margins must be non-negative")
        return left_margin, right_margin

    def set_legend_axis(self, axis_name: str | None) -> Self:
        """
        Set the axis where legend should be displayed for detection components.

        Parameters
        ----------
        axis_name
            Name of the axis for legend display. None means auto (timeseries).

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._legend_axis = axis_name
        return self

    def _create_default_visualizers(self) -> None:
        """Create default visualizer instances with tutorial-matching defaults."""
        # Create timeseries visualizer
        self.timeseries_visualizer = UnivariateTimeseriesVisualizer(backend=self._backend)
        if self._data_provider is not None:
            self.timeseries_visualizer.set_data_provider(self._data_provider)

        # Set timeseries defaults to match tutorial
        self.timeseries_visualizer.set_plot_opts(
            title="Time Series with Change Points",
            xlabel="Time Index",
            ylabel="Value",
            grid=True,
        ).set_draw_opts(
            color="black",
            linewidth=1.5,
            alpha=0.7,
            label="Time Series",
        )

        # Create trace visualizer
        state_visualizer = DummyStateVisualizer[StateT](backend=self._backend)
        self.trace_visualizer = OnlineTraceVisualizer[StateT](backend=self._backend, state_visualizer=state_visualizer)
        if self._detection_trace is not None:
            self.trace_visualizer.set_trace(self._detection_trace)

        # Set trace visualizer defaults to match tutorial
        self.trace_visualizer.set_detection_func_plot_opts(
            title="Detection Function",
            xlabel="Time Index",
            ylabel="Detection Statistic",
            grid=True,
        ).set_detection_func_draw_opts(
            color="blue",
            linewidth=1,
            label="Detection Function",
        ).set_threshold_draw_opts(
            color="red",
            linestyle="dash",
            linewidth=2,
            alpha=0.8,
            label="Threshold",
        ).set_processing_time_plot_opts(
            title="Processing Time per Step",
            xlabel="Time Index",
            ylabel="Time (seconds)",
            grid=True,
        ).set_processing_time_draw_opts(
            color="purple",
            linewidth=1,
            fill_alpha=0.3,
            label="Processing Time",
        )

    def _create_figure_and_axes(self) -> tuple[Figure, AxMapping]:
        """Create figure and axes using fixed or automatic layouts."""
        if isinstance(self._layout, VerticalLayout):
            return self._create_automatic_vertical_layout()
        if isinstance(self._layout, SplitLayout):
            return self._create_automatic_split_layout()
        return self._layout.create_figure_and_axes()

    def _ordered_timeseries_axes(self) -> list[str]:
        """Return timeseries axes in stable visual order."""
        if self.timeseries_visualizer is None:
            return []
        ordered_axes = getattr(self.timeseries_visualizer, "ordered_axes", None)
        if isinstance(ordered_axes, list):
            return ordered_axes
        axes = list(self.timeseries_visualizer.axes)
        timeseries_axes = [axis_name for axis_name in axes if axis_name.startswith("timeseries_")]
        other_axes = [axis_name for axis_name in axes if not axis_name.startswith("timeseries_")]
        timeseries_axes.sort(key=self._axis_sort_key)
        other_axes.sort()
        return timeseries_axes + other_axes

    @staticmethod
    def _axis_sort_key(axis_name: str) -> tuple[int, str]:
        if axis_name.startswith("timeseries_"):
            suffix = axis_name.removeprefix("timeseries_")
            if suffix.isdigit():
                return int(suffix), axis_name
        return 10**9, axis_name

    def _ordered_trace_axes(self) -> list[str]:
        ordered_axes: list[str] = []
        if self.trace_visualizer is None:
            return ordered_axes
        for axis_name in ("detection_function", "processing_time"):
            if axis_name in self.trace_visualizer.axes:
                ordered_axes.append(axis_name)
        state_axes = sorted(
            axis_name for axis_name in self.trace_visualizer.state_visualizer.axes if axis_name not in ordered_axes
        )
        ordered_axes.extend(state_axes)
        other_axes = sorted(axis_name for axis_name in self.trace_visualizer.axes if axis_name not in ordered_axes)
        ordered_axes.extend(other_axes)
        return ordered_axes

    def _axes_with_secondary_y(self) -> set[str]:
        axes_with_twin = getattr(self.timeseries_visualizer, "axes_with_twin", None)
        return axes_with_twin if isinstance(axes_with_twin, set) else set()

    @staticmethod
    def _default_axis_title(axis_name: str) -> str:
        if axis_name == "timeseries":
            return "Time Series with Change Points"
        if axis_name == "detection_function":
            return "Detection Function"
        if axis_name == "processing_time":
            return "Processing Time per Step"
        return axis_name.replace("_", " ").title()

    def _split_titles(self, rows: int, timeseries_axes: list[str], trace_axes: list[str]) -> tuple[str, ...]:
        titles: list[str] = []
        for row_index in range(rows):
            left_title = (
                self._default_axis_title(timeseries_axes[row_index]) if row_index < len(timeseries_axes) else ""
            )
            right_title = self._default_axis_title(trace_axes[row_index]) if row_index < len(trace_axes) else ""
            titles.extend([left_title, right_title])
        return tuple(titles)

    def _create_automatic_vertical_layout(self) -> tuple[Figure, AxMapping]:
        ordered_axes = self._ordered_timeseries_axes() + self._ordered_trace_axes()
        if not ordered_axes:
            ordered_axes = ["timeseries"]
        if self.backend == DrawBackend.PLOTLY:
            specs = [[{"secondary_y": axis_name in self._axes_with_secondary_y()}] for axis_name in ordered_axes]
            figure = make_subplots(
                rows=len(ordered_axes),
                cols=1,
                specs=specs,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=tuple(self._default_axis_title(axis_name) for axis_name in ordered_axes),
            )
            plotly_axes: GoAxMapping = {axis_name: (index + 1, 1) for index, axis_name in enumerate(ordered_axes)}
            return figure, plotly_axes
        figure, mpl_axes = plt.subplots(
            len(ordered_axes),
            1,
            figsize=(14, max(4 * len(ordered_axes), 4)),
            sharex=True,
            squeeze=False,
        )
        matplotlib_axes: PltAxMapping = {axis_name: mpl_axes[index][0] for index, axis_name in enumerate(ordered_axes)}
        return figure, matplotlib_axes

    def _create_automatic_split_layout(self) -> tuple[Figure, AxMapping]:
        timeseries_axes = self._ordered_timeseries_axes()
        trace_axes = self._ordered_trace_axes()
        rows = max(len(timeseries_axes), len(trace_axes), 1)
        if self.backend == DrawBackend.PLOTLY:
            specs = []
            secondary_y_axes = self._axes_with_secondary_y()
            for row_index in range(rows):
                left_spec = (
                    {"secondary_y": timeseries_axes[row_index] in secondary_y_axes}
                    if row_index < len(timeseries_axes)
                    else None
                )
                right_spec = (
                    {"secondary_y": trace_axes[row_index] in secondary_y_axes} if row_index < len(trace_axes) else None
                )
                specs.append([left_spec, right_spec])
            figure = make_subplots(
                rows=rows,
                cols=2,
                specs=specs,
                shared_xaxes=True,
                vertical_spacing=0.08,
                horizontal_spacing=0.1,
                subplot_titles=self._split_titles(rows, timeseries_axes, trace_axes),
            )
            plotly_axes: GoAxMapping = {}
            for index, axis_name in enumerate(timeseries_axes):
                plotly_axes[axis_name] = (index + 1, 1)
            for index, axis_name in enumerate(trace_axes):
                plotly_axes[axis_name] = (index + 1, 2)
            return figure, plotly_axes
        figure, mpl_axes = plt.subplots(rows, 2, figsize=(16, max(4 * rows, 4)), sharex="col", squeeze=False)
        matplotlib_axes: PltAxMapping = {}
        for index, axis_name in enumerate(timeseries_axes):
            matplotlib_axes[axis_name] = mpl_axes[index][0]
        for index, axis_name in enumerate(trace_axes):
            matplotlib_axes[axis_name] = mpl_axes[index][1]
        for index in range(len(timeseries_axes), rows):
            mpl_axes[index][0].set_visible(False)
        for index in range(len(trace_axes), rows):
            mpl_axes[index][1].set_visible(False)
        return figure, matplotlib_axes

    def _create_default_components(self) -> None:
        """Create default visual components for change-point visualizers."""
        # Ground truth lines (requires set_ground_truth to have data)
        self._components["ground_truth_lines"] = (
            VerticalLineVisualComponent(self._backend).set_style(
                color="red", linestyle="solid", linewidth=2, alpha=0.8, label="Ground Truth", legend=True
            ),
            ["timeseries", "detection_function"],
            True,
        )

        # Margin fill (requires set_ground_truth with margin > 0)
        self._components["margin_fill"] = (
            VerticalFillComponent(self._backend).set_style(
                fill_color="red", fill_alpha=0.1, label="Margin Window", legend=True
            ),
            ["timeseries"],
            True,
        )

        # Detected change points (from trace.signal_change_points)
        self._components["detected_lines"] = (
            VerticalLineVisualComponent(self._backend).set_style(
                color="green", linestyle="dash", linewidth=2, alpha=0.8, label="Detected CP", legend=True
            ),
            ["timeseries", "detection_function"],
            True,
        )

        # Forced change points (from trace.forced_change_points)
        self._components["forced_lines"] = (
            VerticalLineVisualComponent(self._backend).set_style(
                color="orange", linestyle="dash", linewidth=2, alpha=0.8, label="Forced CP", legend=True
            ),
            ["timeseries", "detection_function"],
            True,
        )

        # Skip periods (from trace.skip_periods)
        self._components["skip_fill"] = (
            VerticalFillComponent(self._backend).set_style(
                fill_color="brown", fill_alpha=0.2, label="Skip Period", legend=True
            ),
            ["timeseries", "detection_function", "processing_time"],
            True,
        )

        # Learning periods (from trace.learning_periods)
        self._components["learning_fill"] = (
            VerticalFillComponent(self._backend).set_style(
                fill_color="green", fill_alpha=0.2, label="Learning Period", legend=True
            ),
            ["timeseries", "detection_function", "processing_time"],
            True,
        )

    def add_component(
        self,
        name: str,
        component: IVisualComponent,
        axes_names: list[str],
        show_legend: bool = True,
    ) -> Self:
        """
        Add a visual component to the plotter.

        Parameters
        ----------
        name
            Name to identify the component.
        component
            Component instance to add.
        axes_names
            Names of axes where the component should be drawn. Must be provided.
        show_legend
            Whether to show legend for this component.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._components[name] = (component, axes_names, show_legend)
        return self

    def remove_component(self, name: str) -> Self:
        """
        Remove a component from the plotter.

        Parameters
        ----------
        name
            Name of the component to remove.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        if name in self._components:
            del self._components[name]
        return self

    def get_component(self, name: str) -> IVisualComponent | None:
        """
        Get a component by name.

        Parameters
        ----------
        name
            Name of the component to retrieve.

        Returns
        -------
        IVisualComponent | None
            The component instance or None if not found.
        """
        component_data = self._components.get(name)
        if component_data:
            return component_data[0]
        return None

    def draw(self, figure: Figure, axes: AxMapping) -> Figure:
        """
        Coordinate drawing of all visualizers and components.

        Parameters
        ----------
        figure
            The figure to draw on (Matplotlib or Plotly).
        axes
            Mapping from subplot names to their axes objects or positions.

        Returns
        -------
        Figure
            The modified figure with all visual elements drawn.
        """
        # Draw all visualizers
        if self.timeseries_visualizer is not None:
            figure = self.timeseries_visualizer.draw(figure=figure, axes=axes)

        if self.trace_visualizer is not None:
            figure = self.trace_visualizer.draw(figure=figure, axes=axes)

        # Determine legend display axis (default: timeseries for detection components)
        legend_axis = self._legend_axis or "timeseries"

        # Draw all components
        for name, (component, axes_names, show_legend) in self._components.items():
            # Set data based on component name
            if name == "detected_lines" and self._detection_trace:
                if hasattr(component, "set_lines"):
                    component.set_lines(self._detection_trace.signal_change_points)
            elif name == "forced_lines" and self._detection_trace:
                if hasattr(component, "set_lines"):
                    component.set_lines(self._detection_trace.forced_change_points)
            elif name == "skip_fill" and self._detection_trace:
                if hasattr(component, "set_regions"):
                    component.set_regions(self._detection_trace.skip_periods)
            elif name == "learning_fill" and self._detection_trace and hasattr(component, "set_regions"):
                component.set_regions(self._detection_trace.learning_periods)

            target_axes_names = self._expand_component_axes_names(axes_names)
            if name in ("skip_fill", "learning_fill") and self.trace_visualizer is not None:
                for state_axis_name in self.trace_visualizer.state_visualizer.axes:
                    if state_axis_name not in target_axes_names:
                        target_axes_names.append(state_axis_name)

            legend_target_axes = self._expand_component_axes_names([legend_axis])
            legend_axes = [axis_name for axis_name in legend_target_axes if axis_name in target_axes_names]
            legend_axis_name = legend_axes[0] if legend_axes else target_axes_names[0]

            # Draw on all specified axes, but emit only one legend entry per component.
            for axes_name in target_axes_names:
                if axes_name in axes:
                    draw_legend = show_legend and axes_name == legend_axis_name
                    component.draw(figure, axes[axes_name], add_legend=draw_legend)

        return figure

    def _expand_component_axes_names(self, axes_names: list[str]) -> list[str]:
        expanded: list[str] = []
        timeseries_axes = self._ordered_timeseries_axes()
        for axes_name in axes_names:
            if axes_name == "timeseries" and timeseries_axes:
                for timeseries_axis_name in timeseries_axes:
                    if timeseries_axis_name not in expanded:
                        expanded.append(timeseries_axis_name)
                continue
            if axes_name not in expanded:
                expanded.append(axes_name)
        return expanded

    @property
    def required_axes(self) -> set[str]:
        """
        Return the set of all axes names required by visualizers and components.

        Returns
        -------
        set[str]
            Set of subplot names needed for drawing.
        """
        axes_set: set[str] = set()

        # Add axes from visualizers
        if self.timeseries_visualizer is not None:
            axes_set.update(self.timeseries_visualizer.axes)
        if self.trace_visualizer is not None:
            axes_set.update(self.trace_visualizer.axes)

        # Add axes from components
        for _, axes_names, _ in self._components.values():
            axes_set.update(axes_names)

        if not isinstance(self._layout, VerticalLayout | SplitLayout):
            axes_set.update(self._layout.required_axes)

        return axes_set

    def set_layout(self, layout: ILayoutStrategy | str) -> Self:
        """
        Set the layout strategy for this plotter.

        Parameters
        ----------
        layout
            Layout strategy to use. Can be:

            - An ILayoutStrategy instance
            - One of: "vertical", "split", "dashboard-lite", "dashboard"

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._layout = self._resolve_layout(layout)
        return self
