# -*- coding: ascii -*-
"""
Visualization interfaces for change-point detection results.

This module defines abstract source classes for visualizers that render
time series data, detection traces, and algorithm performance metrics
using either Matplotlib or Plotly backends.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod
from typing import overload

from pysatl_cpd.analysis.visualization.typedefs import (
    AxMapping,
    DrawBackend,
    Figure,
    GoAxMapping,
    GoFigure,
    PltAxMapping,
    PltFigure,
)
from pysatl_cpd.analysis.visualization.utils import (
    is_matplotlib_figure,
    is_matplotlib_mapping,
    is_plotly_figure,
    is_plotly_mapping,
    normalize_backend,
)


class IVisualizer(ABC):
    """
    Abstract source class for all visualizers.

    This interface defines the contract for visualizers that render data
    onto specific subplots within a figure. Each visualizer declares which
    subplot axes it requires and provides backend-specific drawing methods.

    The visualizer pattern enables composable figure construction, where
    a coordinator creates a figure with named subplots, and each visualizer
    draws its content onto its designated axes.

    Parameters
    ----------
    backend
        Plotting backend to use for rendering (MATPLOTLIB or PLOTLY).
    """

    def __init__(self, backend: DrawBackend | str) -> None:
        self._backend = normalize_backend(backend)

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
        """Set the plotting backend.

        Parameters
        ----------
        value
            Backend name (``MATPLOTLIB`` or ``PLOTLY``).
        """
        self._backend = normalize_backend(value)

    @property
    @abstractmethod
    def axes(self) -> set[str]:  # pragma: no cover
        """
        Declare the subplot names required by this visualizer.

        Returns
        -------
        set[str]
            Set of subplot identifiers that this visualizer will draw onto.
            These names must correspond to axes provided in the AxMapping
            when the draw method is called.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this property.
        """
        raise NotImplementedError

    @overload
    def draw(self, *, figure: PltFigure, axes: PltAxMapping) -> PltFigure: ...  # pragma: no cover

    @overload
    def draw(self, *, figure: GoFigure, axes: GoAxMapping) -> GoFigure: ...  # pragma: no cover

    def draw(self, *, figure: Figure, axes: AxMapping) -> Figure:
        """
        Draw content onto the specified axes.

        This method dispatches to the appropriate backend-specific
        implementation based on the backend argument. The figure is
        inferred from the axes objects (for Matplotlib) or created
        automatically (for Plotly).

        Parameters
        ----------
        figure
            The figure to draw onto (Matplotlib or Plotly).
        axes
            Mapping from subplot names to their axes objects or positions.

        Returns
        -------
        Figure
            The modified figure object.

        Raises
        ------
        TypeError
            If the figure or axes mapping type does not match the backend.
        ValueError
            If the backend is not supported.
        """
        if self.backend == DrawBackend.PLOTLY:
            if not is_plotly_figure(figure):
                raise TypeError(f"Expected GoFigure for Plotly backend, got {type(figure)}")
            if not is_plotly_mapping(axes):
                raise TypeError(f"Expected Plotly axes mapping for Plotly backend, got {type(axes)}")
            return self._draw_plotly(figure, axes)

        if self.backend == DrawBackend.MATPLOTLIB:
            if not is_matplotlib_figure(figure):
                raise TypeError(f"Expected PltFigure for Matplotlib backend, got {type(figure)}")
            if not is_matplotlib_mapping(axes):
                raise TypeError(f"Expected Matplotlib axes mapping for Matplotlib backend, got {type(axes)}")
            return self._draw_matplotlib(figure, axes)

        # This line should never be reached due to validation in __init__
        raise ValueError(f"Unsupported backend: {self.backend}")

    @abstractmethod
    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxMapping) -> PltFigure:  # pragma: no cover
        """
        Draw using Matplotlib backend.

        Parameters
        ----------
        figure
            The Matplotlib figure to draw onto.
        axes
            Mapping from subplot names to Matplotlib axes objects.

        Returns
        -------
        PltFigure
            The figure containing the drawn axes.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def _draw_plotly(self, figure: GoFigure, axes: GoAxMapping) -> GoFigure:  # pragma: no cover
        """
        Draw using Plotly backend.

        Parameters
        ----------
        figure
            The Plotly figure to draw onto.
        axes
            Mapping from subplot names to Plotly subplot positions.

        Returns
        -------
        GoFigure
            The modified Plotly figure.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError
