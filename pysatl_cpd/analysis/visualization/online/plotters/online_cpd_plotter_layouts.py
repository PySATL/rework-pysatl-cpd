# -*- coding: ascii -*-
"""
Layout strategies for OnlineCPDPlotter.

This module provides various layout strategies for arranging subplots
in change-point detection visualizations.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod
from collections.abc import Callable

from plotly.subplots import make_subplots

from pysatl_cpd.analysis.visualization.typedefs import AxMapping, Figure


class ILayoutStrategy(ABC):
    """Abstract base class for plot layout strategies."""

    @abstractmethod
    def create_figure_and_axes(self) -> tuple[Figure, AxMapping]:  # pragma: no cover
        """Create a figure and return its axes mapping."""
        raise NotImplementedError

    @property
    @abstractmethod
    def required_axes(self) -> set[str]:  # pragma: no cover
        """Return the subplot names provided by this layout."""
        raise NotImplementedError


class VerticalLayout(ILayoutStrategy):
    """Stack all plots in a single column."""

    def create_figure_and_axes(self) -> tuple[Figure, AxMapping]:
        """Create a three-row vertical figure."""
        figure = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Time Series with Change Points", "Detection Function", "Processing Time per Step"),
        )

        axes: AxMapping = {
            "timeseries": (1, 1),
            "detection_function": (2, 1),
            "processing_time": (3, 1),
        }

        return figure, axes

    @property
    def required_axes(self) -> set[str]:
        """Return axes required by vertical layout."""
        return {"timeseries", "detection_function", "processing_time"}


class SplitLayout(ILayoutStrategy):
    """Put the time series in one column and trace plots in the other."""

    def create_figure_and_axes(self) -> tuple[Figure, AxMapping]:
        """Create a three-row, two-column split figure."""
        figure = make_subplots(
            rows=3,
            cols=2,
            shared_xaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
            subplot_titles=(
                "Time Series with Change Points",
                "Detection Function",
                "Processing Time per Step",
                "",  # Empty for state plot
                "",  # Empty for additional plots
                "",  # Empty for additional plots
            ),
        )

        axes: AxMapping = {
            "timeseries": (1, 1),
            "detection_function": (1, 2),
            "processing_time": (2, 2),
            "state": (3, 2),
        }

        return figure, axes

    @property
    def required_axes(self) -> set[str]:
        """Return axes required by split layout."""
        return {"timeseries", "detection_function", "processing_time", "state"}


class DashboardLiteLayout(ILayoutStrategy):
    """Show one time-series panel above two trace panels."""

    def create_figure_and_axes(self) -> tuple[Figure, AxMapping]:
        """Create a compact two-row dashboard figure."""
        figure = make_subplots(
            rows=2,
            cols=2,
            shared_xaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
            subplot_titles=(
                "Time Series with Change Points",
                "",  # Empty for state plot
                "Detection Function",
                "Processing Time per Step",
            ),
            column_widths=[0.7, 0.3],
        )

        axes: AxMapping = {
            "timeseries": (1, 1),
            "state": (1, 2),
            "detection_function": (2, 1),
            "processing_time": (2, 2),
        }

        return figure, axes

    @property
    def required_axes(self) -> set[str]:
        """Return axes required by dashboard-lite layout."""
        return {"timeseries", "detection_function", "processing_time", "state"}


class DashboardLayout(ILayoutStrategy):
    """Arrange the time series and three trace plots in a 2x2 grid."""

    def create_figure_and_axes(self) -> tuple[Figure, AxMapping]:
        """Create a full two-by-two dashboard figure."""
        figure = make_subplots(
            rows=2,
            cols=2,
            shared_xaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
            subplot_titles=(
                "Time Series with Change Points",
                "State Evolution",
                "Detection Function",
                "Processing Time per Step",
            ),
        )

        axes: AxMapping = {
            "timeseries": (1, 1),
            "state": (1, 2),
            "detection_function": (2, 1),
            "processing_time": (2, 2),
        }

        return figure, axes

    @property
    def required_axes(self) -> set[str]:
        """Return axes required by dashboard layout."""
        return {"timeseries", "detection_function", "processing_time", "state"}


class CustomLayout(ILayoutStrategy):
    """Allow callers to provide custom figure-construction logic."""

    def __init__(self, create_figure_func: Callable[[], tuple[Figure, AxMapping]], required_axes: set[str]) -> None:
        self._create_figure_func = create_figure_func
        self._required_axes = required_axes

    def create_figure_and_axes(self) -> tuple[Figure, AxMapping]:
        """Create a figure with the user-provided factory."""
        return self._create_figure_func()

    @property
    def required_axes(self) -> set[str]:
        """Return axes required by custom layout."""
        return self._required_axes
