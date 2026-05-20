# -*- coding: ascii -*-
"""
Abstract visualizer interfaces for the PySATL CPD visualization system.

This subpackage exports the abstract base classes that define the contracts
for visualizers and composable drawing components. Concrete implementations
subclass these interfaces to render time series data, detection traces,
benchmark metrics, and annotation elements using either Matplotlib or Plotly
backends.

.. raw:: html

    <h2>Architecture</h2>

The visualization layer implements two complementary design patterns:

- **Strategy Pattern** (``IVisualizer`` hierarchy): Visualizers own complete
  subplots and encapsulate the drawing logic for a specific data type. Backend
  selection determines which drawing strategy is invoked through the template
  method pattern.
- **Component Pattern** (``IVisualComponent``): Components add discrete visual
  elements (change-point markers, period fills, annotation lines) to existing
  subplots. Multiple components can be composed on the same subplot.

Separation of concerns is strict: visualizers configure axes properties and
draw primary data content; components draw supplementary annotations; a
coordinator (outside this subpackage) creates the figure layout and orchestrates
drawing order.

.. raw:: html

    <h2>Public API</h2>

- ``IVisualizer``: Base interface for all visualizers that manage complete
  subplots. Declares ``axes``, ``backend``, and backend-specific ``draw``
  methods.
- ``ITimeseriesVisualizer``: Interface for time series visualizers. Generic
  over ``DataProvider``; requires subclasses to implement ``set_data_provider``.
- ``ITraceVisualizer``: Interface for detection trace visualizers. Generic
  over ``DetectionTrace``; requires subclasses to implement ``set_trace``.
- ``IMetricVisualizer``: Base class for benchmark metric visualizers. Manages
  benchmark tables and per-entry line styling.
- ``IVisualComponent``: Interface for composable drawing components. Renders
  specific annotation elements onto a single subplot with optional legend
  entries.

See each class's own docstring for full parameter and method details.

.. raw:: html

    <h2>Backend Abstraction</h2>

All visualizers and components implement both ``_draw_matplotlib()`` and
``_draw_plotly()`` methods. The public ``draw()`` method dispatches to the
appropriate backend-specific implementation based on the ``backend`` property.
Backends are specified via the ``DrawBackend`` enum (``MATPLOTLIB`` or
``PLOTLY``).

.. raw:: html

    <h2>Legend Management</h2>

Components store legend state in their shared visual specs (``label``,
``legend``, ``legend_group``). When ``add_legend=True`` is passed to
``draw()``, the component may add its legend entry if the element spec also
enables legend. For Plotly, legend entries are grouped by effective legend
group, defaulting to the element label.

Examples
--------
Creating a concrete visualizer by subclassing ``IVisualizer``::

    >>> import matplotlib.pyplot as plt
    >>> from pysatl_cpd.analysis.visualization.abstracts import IVisualizer
    >>> from pysatl_cpd.analysis.visualization.typedefs import (
    ...     DrawBackend,
    ...     GoAxMapping,
    ...     GoFigure,
    ...     PltAxMapping,
    ...     PltFigure,
    ... )
    >>> class SimpleScatterVisualizer(IVisualizer):
    ...     def __init__(self, backend: DrawBackend = DrawBackend.MATPLOTLIB):
    ...         super().__init__(backend)
    ...         self._x: list[float] = []
    ...         self._y: list[float] = []
    ...     @property
    ...     def axes(self) -> set[str]:
    ...         return {"scatter"}
    ...     def set_data(self, x: list[float], y: list[float]) -> None:
    ...         self._x = x
    ...         self._y = y
    ...     def _draw_matplotlib(self, figure: PltFigure, axes: PltAxMapping) -> PltFigure:
    ...         ax = axes["scatter"]
    ...         ax.scatter(self._x, self._y)
    ...         return figure
    ...     def _draw_plotly(self, figure: GoFigure, axes: GoAxMapping) -> GoFigure:
    ...         import plotly.graph_objects as go
    ...         row, col = axes["scatter"]
    ...         figure.add_trace(go.Scatter(x=self._x, y=self._y, mode="markers"), row=row, col=col)
    ...         return figure
    >>> viz = SimpleScatterVisualizer(DrawBackend.MATPLOTLIB)
    >>> viz.set_data([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    >>> print(viz.axes)
    {'scatter'}

Creating a concrete component by subclassing ``IVisualComponent``::

    >>> import matplotlib.pyplot as plt
    >>> from pysatl_cpd.analysis.visualization.abstracts import IVisualComponent
    >>> from pysatl_cpd.analysis.visualization.typedefs import (
    ...     DrawBackend,
    ...     GoAxes,
    ...     GoFigure,
    ...     PltAxes,
    ...     PltFigure,
    ... )
    >>> class HorizontalLineComponent(IVisualComponent):
    ...     def __init__(self, backend: DrawBackend = DrawBackend.MATPLOTLIB):
    ...         super().__init__(backend)
    ...         self._y_value: float = 0.0
    ...     def set_y(self, y: float) -> None:
    ...         self._y_value = y
    ...     def _draw_matplotlib(self, figure: PltFigure, axes: PltAxes, add_legend: bool = False) -> None:
    ...         axes.axhline(self._y_value, color="red", linestyle="--")
    ...     def _draw_plotly(self, figure: GoFigure, axes: GoAxes, add_legend: bool = False) -> None:
    ...         row, col = axes
    ...         figure.add_hline(y=self._y_value, line_color="red", line_dash="dash", row=row, col=col)
    >>> comp = HorizontalLineComponent(DrawBackend.MATPLOTLIB)
    >>> comp.set_y(0.5)
    >>> fig, ax = plt.subplots()
    >>> comp.draw(fig, ax)
    >>> plt.close(fig)

Switching backends on an existing visualizer or component::

    >>> from pysatl_cpd.analysis.visualization.abstracts import IVisualComponent
    >>> from pysatl_cpd.analysis.visualization.typedefs import DrawBackend
    >>> comp = IVisualComponent.__new__(IVisualComponent)
    >>> # Concrete components expose a backend property that can be reassigned:
    >>> # comp.backend = DrawBackend.PLOTLY

Notes
-----
- All classes in this subpackage are abstract. They cannot be instantiated
  directly; concrete subclasses must implement the required ``_draw_matplotlib``
  and ``_draw_plotly`` methods.
- Type parameters on ``ITimeseriesVisualizer`` and ``ITraceVisualizer`` use
  Python 3.12+ PEP 695 syntax. The module requires Python 3.12 or later.
- Matplotlib is required for the ``MATPLOTLIB`` backend; plotly is required
  for the ``PLOTLY`` backend. Both are listed as optional dependencies in
  ``pyproject.toml``.
- Change-point indices are zero-based throughout the visualization system.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.analysis.visualization.abstracts.imetric_visualizer import IMetricVisualizer
from pysatl_cpd.analysis.visualization.abstracts.itimeseries_visualizer import (
    ITimeseriesVisualizer,
)
from pysatl_cpd.analysis.visualization.abstracts.itrace_visualizer import ITraceVisualizer
from pysatl_cpd.analysis.visualization.abstracts.ivisual_component import IVisualComponent
from pysatl_cpd.analysis.visualization.abstracts.ivisualizer import IVisualizer

__all__ = [
    "IVisualizer",
    "ITimeseriesVisualizer",
    "ITraceVisualizer",
    "IVisualComponent",
    "IMetricVisualizer",
]
