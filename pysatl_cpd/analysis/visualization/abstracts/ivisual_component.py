# -*- coding: ascii -*-
"""
Visualization component interfaces.

This module defines the abstract source class for composable visualizers
components that draw specific elements (change points, segments, annotations)
onto a single subplot. Components are backend-agnostic and provide separate
implementations for Matplotlib and Plotly backends.

Components must support optional legend entries with the ability to toggle
visibility interactively in Plotly through legend groups.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from typing import overload

from pysatl_cpd.analysis.visualization.typedefs import DrawBackend, Figure, GoAxes, GoFigure, PltAxes, PltFigure
from pysatl_cpd.analysis.visualization.utils import normalize_backend


class IVisualComponent(ABC):
    """
    Base interface for composable visualizers components.

    Components are responsible for drawing specific elements (e.g., change points,
    segments, annotations) onto a single subplot. They are backend-agnostic and
    implement both Matplotlib and Plotly drawing methods.

    Parameters
    ----------
    backend
        The backend to use for drawing operations.
    """

    def __init__(self, backend: DrawBackend | str) -> None:
        self._backend = normalize_backend(backend)

    @property
    def backend(self) -> DrawBackend:
        """Return the current drawing backend."""
        return self._backend

    @backend.setter
    def backend(self, value: DrawBackend | str) -> None:
        self._backend = normalize_backend(value)

    @overload
    def draw(self, figure: PltFigure, axes: PltAxes, add_legend: bool = False) -> None:
        """
        Draw component on a Matplotlib axes.

        Parameters
        ----------
        figure
            The Matplotlib figure containing the axes.
        axes
            The Matplotlib axes to draw on.
        add_legend
            Whether to add legend entry for this component.
        """

    @overload
    def draw(self, figure: GoFigure, axes: GoAxes, add_legend: bool = False) -> None:
        """
        Draw component on a Plotly subplot.

        Parameters
        ----------
        figure
            The Plotly figure containing the subplot.
        axes
            The subplot position (row, col) to draw on.
        add_legend
            Whether to display legend entry for this component.
        """

    def draw(self, figure: Figure, axes: GoAxes | PltAxes, add_legend: bool = False) -> None:
        """
        Draw component on a subplot.

        Parameters
        ----------
        figure
            The figure containing the subplot (Matplotlib or Plotly).
        axes
            The subplot data to draw on.
        add_legend
            Whether to add (for Matplotlib) or display (for Plotly)
            legend entry for this component.

        Raises
        ------
        TypeError
            If the figure or axes type does not match the current backend.
        ValueError
            If drawing with legend when label is not set, or if an
            unsupported backend is provided.
        """
        style = getattr(self, "_style", None)
        if add_legend and isinstance(style, dict) and style.get("legend", True) and style.get("label") is None:
            raise ValueError("Can not draw with legend: label is set to None")

        if self._backend == DrawBackend.MATPLOTLIB:
            if not isinstance(figure, PltFigure.__value__):
                raise TypeError(f"Expected PltFigure for Matplotlib backend, got {type(figure)}")
            if not isinstance(axes, PltAxes.__value__):
                raise TypeError(f"Expected PltAxes for Matplotlib backend, got {type(axes)}")
            self._draw_matplotlib(figure, axes, add_legend)
        elif self._backend == DrawBackend.PLOTLY:
            if not isinstance(figure, GoFigure.__value__):
                raise TypeError(f"Expected GoFigure for Plotly backend, got {type(figure)}")
            if not (
                isinstance(axes, tuple) and (len(axes) == 2) and isinstance(axes[0], int) and isinstance(axes[1], int)
            ):
                raise TypeError(f"Expected tuple of ints for Plotly backend, got {axes}")
            self._draw_plotly(figure, axes, add_legend)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    @abstractmethod
    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxes, add_legend: bool = False) -> None:  # pragma: no cover
        """
        Backend-specific drawing implementation for Matplotlib.
        """
        raise NotImplementedError

    @abstractmethod
    def _draw_plotly(self, figure: GoFigure, axes: GoAxes, add_legend: bool = False) -> None:  # pragma: no cover
        """
        Backend-specific drawing implementation for Plotly.
        """
        raise NotImplementedError
