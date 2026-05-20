# -*- coding: ascii -*-
"""
State visualizers for online algorithm state evolution.

This module provides the interface and concrete implementations for rendering
the evolution of online change-point detection algorithm state over time.
State visualizers implement the IOnlineStateVisualizer interface and are
designed to be composed with OnlineTraceVisualizer to add algorithm-specific
state panels (e.g., running means, control limits, window statistics) to
detection trace figures.

The module follows the visualization system's composable architecture: each
state visualizer declares its required subplot axes through the ``axes``
property, accepts a sequence of algorithm state snapshots via ``set_states``,
and renders onto Matplotlib or Plotly figures through the backend dispatch
inherited from IVisualizer.

.. raw:: html

    <h2>Public API</h2>

- ``IOnlineStateVisualizer``: Abstract source interface for all online state
  visualizers. Generic over a type parameter bound to ``OnlineAlgorithmState``.
  Requires subclasses to implement ``set_states`` and the backend-specific
  drawing methods.
- ``DummyStateVisualizer``: Placeholder visualizer that performs no rendering.
  Declares an empty ``axes`` set and leaves figures unchanged on draw. Useful
  for testing or when state visualization is not needed.

See each class's own docstring for full parameter and method details.

.. raw:: html

    <h2>Composition With Trace Visualizers</h2>

State visualizers are typically passed to ``OnlineTraceVisualizer`` via the
``state_visualizer`` constructor argument. The trace visualizer merges the
state visualizer's declared axes into the overall figure layout and forwards
collected algorithm states through ``set_states`` during drawing.

Examples
--------
Creating a dummy state visualizer and using it with OnlineTraceVisualizer::

    >>> from pysatl_cpd.analysis.visualization.online import (
    ...     DummyStateVisualizer,
    ...     OnlineTraceVisualizer,
    ... )
    >>> from pysatl_cpd.analysis.visualization.typedefs import DrawBackend
    >>> from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState
    >>> state_viz = DummyStateVisualizer[OnlineAlgorithmState](backend=DrawBackend.MATPLOTLIB)
    >>> trace_viz = OnlineTraceVisualizer(
    ...     backend=DrawBackend.MATPLOTLIB,
    ...     state_visualizer=state_viz,
    ... )
    >>> print(state_viz.axes)
    set()
    >>> print("shewhart_state" in trace_viz.axes)
    False

Creating a concrete state visualizer by subclassing IOnlineStateVisualizer::

    >>> import matplotlib.pyplot as plt
    >>> from pysatl_cpd.analysis.visualization.online.states import IOnlineStateVisualizer
    >>> from pysatl_cpd.analysis.visualization.typedefs import (
    ...     DrawBackend,
    ...     GoAxMapping,
    ...     GoFigure,
    ...     PltAxMapping,
    ...     PltFigure,
    ... )
    >>> from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState
    >>> class MyStateVisualizer(IOnlineStateVisualizer[OnlineAlgorithmState]):
    ...     def __init__(self, backend: DrawBackend = DrawBackend.MATPLOTLIB):
    ...         super().__init__(backend)
    ...         self._states: list[OnlineAlgorithmState | None] = []
    ...     @property
    ...     def axes(self) -> set[str]:
    ...         return {"my_state"}
    ...     def set_states(self, states):
    ...         self._states = list(states)
    ...         return self
    ...     def _draw_matplotlib(self, figure: PltFigure, axes: PltAxMapping) -> PltFigure:
    ...         ax = axes["my_state"]
    ...         ax.set_title("My State")
    ...         return figure
    ...     def _draw_plotly(self, figure: GoFigure, axes: GoAxMapping) -> GoFigure:
    ...         import plotly.graph_objects as go
    ...         row, col = axes["my_state"]
    ...         figure.update_layout(title="My State")
    ...         return figure
    >>> viz = MyStateVisualizer(DrawBackend.MATPLOTLIB)
    >>> print(viz.axes)
    {'my_state'}

Notes
-----
- All state visualizers use Python 3.12+ PEP 695 generic syntax for their
  type parameters. The module requires Python 3.12 or later.
- Matplotlib is required for the ``MATPLOTLIB`` backend; plotly is required
  for the ``PLOTLY`` backend. Both are listed as optional dependencies in
  ``pyproject.toml``.
- Change-point indices are zero-based throughout the visualization system.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.analysis.visualization.online.states.ionline_state_visualizer import (
    IOnlineStateVisualizer,
)
from pysatl_cpd.analysis.visualization.online.states.state_dummy_visualizer import (
    DummyStateVisualizer,
)

__all__ = [
    "IOnlineStateVisualizer",
    "DummyStateVisualizer",
]
