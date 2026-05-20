# -*- coding: ascii -*-
"""
Reusable visualization components for vertical annotations.

This module provides backend-agnostic, composable drawing components that
render vertical lines and filled vertical regions onto Matplotlib axes or
Plotly subplots. Components implement the ``IVisualComponent`` interface
(see ``pysatl_cpd.analysis.visualization.abstracts``) and are designed to
be layered on top of subplots created by higher-level visualizers.

Each component supports configurable styling via ``set_style()`` and
method chaining through ``set_lines()``, ``set_regions()``, and ``clear()``.

.. raw:: html

    <h2>Public API</h2>

- ``VerticalLineVisualComponent`` -- draws vertical lines at specified
  x-coordinates. Style is configured via ``LineSpec`` parameters
  (color, linestyle, linewidth, alpha, label, legend).
- ``VerticalFillComponent`` -- draws filled vertical regions between
  pairs of x-coordinates. Style is configured via ``FillSpec`` parameters
  (fill_color, fill_alpha, label, legend).

Examples
--------

Add vertical lines and fills to a Matplotlib figure:

>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> from pysatl_cpd.analysis.visualization import DrawBackend
>>> from pysatl_cpd.analysis.visualization.components import (
...     VerticalFillComponent,
...     VerticalLineVisualComponent,
... )
>>> fig, ax = plt.subplots()
>>> _ = ax.plot(np.arange(10), np.arange(10), label="data")
>>> line_component = (
...     VerticalLineVisualComponent(DrawBackend.MATPLOTLIB)
...     .set_lines([3, 7])
...     .set_style(color="red", linestyle="--", linewidth=2, label="change points", legend=True)
... )
>>> fill_component = (
...     VerticalFillComponent(DrawBackend.MATPLOTLIB)
...     .set_regions([(2, 4), (6, 8)])
...     .set_style(fill_color="blue", fill_alpha=0.15, label="margin", legend=True)
... )
>>> fill_component.draw(fig, ax, add_legend=True)
>>> line_component.draw(fig, ax, add_legend=True)
>>> _ = ax.legend()

Add the same components to a Plotly figure:

>>> import plotly.graph_objects as go
>>> from plotly.subplots import make_subplots
>>> from pysatl_cpd.analysis.visualization import DrawBackend
>>> from pysatl_cpd.analysis.visualization.components import (
...     VerticalFillComponent,
...     VerticalLineVisualComponent,
... )
>>> fig = make_subplots(rows=1, cols=1)
>>> _ = fig.add_trace(go.Scatter(x=list(range(10)), y=list(range(10)), name="data"), row=1, col=1)
>>> line_component = (
...     VerticalLineVisualComponent(DrawBackend.PLOTLY)
...     .set_lines([3, 7])
...     .set_style(color="red", linestyle="dash", linewidth=2, label="change points", legend=True)
... )
>>> fill_component = (
...     VerticalFillComponent(DrawBackend.PLOTLY)
...     .set_regions([(2, 4), (6, 8)])
...     .set_style(fill_color="blue", fill_alpha=0.15, label="margin", legend=True)
... )
>>> fill_component.draw(fig, (1, 1), add_legend=True)
>>> line_component.draw(fig, (1, 1), add_legend=True)

Notes
-----

- Components are backend-agnostic. The same component instance can switch
  between Matplotlib and Plotly by reassigning its ``backend`` property.
- For Plotly, the ``draw()`` method expects a ``(row, col)`` tuple as the
  axes argument, matching the subplot grid created by ``plotly.subplots.make_subplots``.
- Style parameters (``LineSpec``, ``FillSpec``) are defined in
  ``pysatl_cpd.analysis.visualization.specs``.
- This module depends on ``matplotlib`` and ``plotly``.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.analysis.visualization.components.vert_fill import VerticalFillComponent
from pysatl_cpd.analysis.visualization.components.vert_line import VerticalLineVisualComponent

__all__ = [
    "VerticalLineVisualComponent",
    "VerticalFillComponent",
]
