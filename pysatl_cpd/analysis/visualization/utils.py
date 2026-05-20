# -*- coding: ascii -*-
"""
Utility functions for visualizers module.

This module provides helper functions for common visualizers tasks,
including backend-agnostic style translation and figure manipulation.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Any, TypeGuard

import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go

from pysatl_cpd.analysis.visualization.specs import FilledLineSpec, FillSpec, LineSpec, PlotSpec
from pysatl_cpd.analysis.visualization.typedefs import (
    Axes,
    AxMapping,
    DrawBackend,
    Figure,
    GoAxes,
    GoAxMapping,
    GoFigure,
    PltAxes,
    PltAxMapping,
    PltFigure,
)

PLOTLY_TO_MPL_LINESTYLE = {
    "solid": "-",
    "dash": "--",
    "dot": ":",
    "dashdot": "-.",
    "longdash": "--",
    "longdashdot": "-.",
}

MPL_TO_PLOTLY_DASH = {
    "-": "solid",
    "--": "dash",
    ":": "dot",
    "-.": "dashdot",
}

TAB_COLOR_TO_HEX = {
    "tab:blue": "#1f77b4",
    "tab:orange": "#ff7f0e",
    "tab:green": "#2ca02c",
    "tab:red": "#d62728",
    "tab:purple": "#9467bd",
    "tab:brown": "#8c564b",
    "tab:pink": "#e377c2",
    "tab:gray": "#7f7f7f",
    "tab:olive": "#bcbd22",
    "tab:cyan": "#17becf",
}

MPL_TO_PLOTLY_MARKER = {
    ".": "circle",
    ",": "circle",
    "o": "circle",
    "s": "square",
    "^": "triangle-up",
    "v": "triangle-down",
    "<": "triangle-left",
    ">": "triangle-right",
    "d": "diamond",
    "D": "diamond",
    "x": "x",
    "+": "cross",
    "*": "star",
    "p": "pentagon",
    "h": "hexagon",
    "H": "hexagon",
}


def normalize_backend(value: DrawBackend | str) -> DrawBackend:
    """Normalise a backend enum or string to ``DrawBackend``.

    Parameters
    ----------
    value
        Backend as enum or string.

    Returns
    -------
    DrawBackend

    Raises
    ------
    ValueError
        If the string does not match any known backend.
    """
    if isinstance(value, DrawBackend):
        return value
    try:
        return DrawBackend(value)
    except ValueError as exc:
        raise ValueError(f"Unknown backend {value}") from exc


# Figure TypeGuards
def is_plotly_figure(figure: Figure) -> TypeGuard[GoFigure]:
    """Check whether a figure is a Plotly Figure.

    Parameters
    ----------
    figure
        Figure to check.

    Returns
    -------
    TypeGuard[GoFigure]
    """
    return isinstance(figure, go.Figure)


def is_matplotlib_figure(figure: Figure) -> TypeGuard[PltFigure]:
    """Check whether a figure is a Matplotlib Figure.

    Parameters
    ----------
    figure
        Figure to check.

    Returns
    -------
    TypeGuard[PltFigure]
    """
    return isinstance(figure, mpl.figure.Figure)


# Axes TypeGuards
def is_plotly_axes(axes: Axes) -> TypeGuard[GoAxes]:
    """Check whether axes is a Plotly subplot position tuple.

    Parameters
    ----------
    axes
        Axes to check.

    Returns
    -------
    TypeGuard[GoAxes]
    """
    return isinstance(axes, tuple) and len(axes) == 2 and isinstance(axes[0], int) and isinstance(axes[1], int)


def is_matplotlib_axes(axes: Axes) -> TypeGuard[PltAxes]:
    """Check whether axes is a Matplotlib Axes object.

    Parameters
    ----------
    axes
        Axes to check.

    Returns
    -------
    TypeGuard[PltAxes]
    """
    return isinstance(axes, mpl.axes.Axes)


# AxMapping TypeGuards
def is_plotly_mapping(mapping: AxMapping) -> TypeGuard[GoAxMapping]:
    """Check whether mapping contains Plotly subplot positions.

    Parameters
    ----------
    mapping
        Axes mapping to check.

    Returns
    -------
    TypeGuard[GoAxMapping]
    """
    return all(is_plotly_axes(v) for v in mapping.values())


def is_matplotlib_mapping(mapping: AxMapping) -> TypeGuard[PltAxMapping]:
    """Check whether mapping contains Matplotlib Axes objects.

    Parameters
    ----------
    mapping
        Axes mapping to check.

    Returns
    -------
    TypeGuard[PltAxMapping]
    """
    return all(is_matplotlib_axes(v) for v in mapping.values())


def get_plotly_subplot_annotation_index(figure: go.Figure, row: int, col: int) -> int:
    """
    Get the annotation index for a subplot in a Plotly figure.

    This function calculates the correct annotation index for subplot titles
    in Plotly figures with arbitrary grid layouts using row-major ordering.

    Parameters
    ----------
    figure
        The Plotly figure containing subplots
    row
        Subplot row position (1-indexed)
    col
        Subplot column position (1-indexed)

    Returns
    -------
    int
        Annotation index in figure.layout.annotations for the subplot title

    Examples
    --------
    >>> import plotly.graph_objects as go
    >>> fig = go.Figure()
    >>> fig = make_subplots(rows=2, cols=2)
    >>> # Top-left subplot (1,1) -> index 0
    >>> get_plotly_subplot_annotation_index(fig, 1, 1)
    0
    >>> # Bottom-right subplot (2,2) -> index 3
    >>> get_plotly_subplot_annotation_index(fig, 2, 2)
    3
    """
    rows, cols = figure._get_subplot_rows_columns()
    return (row - 1) * max(cols) + (col - 1)  # type: ignore[no-any-return]


def translate_linestyle(linestyle: str) -> str:
    """
    Translate linestyle strings from Plotly to Matplotlib format.

    Parameters
    ----------
    linestyle
        Linestyle string in source format (Plotly).

    Returns
    -------
    str
        Linestyle string in target format.

    Raises
    ------
    ValueError
        If linestyle is not recognized.

    Examples
    --------
    >>> translate_linestyle("dash")
    '--'
    >>> translate_linestyle("dashdot")
    '-.'
    """
    if linestyle in PLOTLY_TO_MPL_LINESTYLE:
        return PLOTLY_TO_MPL_LINESTYLE[linestyle]
    if linestyle in MPL_TO_PLOTLY_DASH:
        return linestyle
    raise ValueError(f"Unrecognized linestyle: {linestyle}")


def to_plotly_color(color: str | None) -> str | None:
    """Convert a Matplotlib-style color to a Plotly-compatible color."""
    if color is None:
        return None
    return TAB_COLOR_TO_HEX.get(color, color)


def to_plotly_dash(linestyle: str | None) -> str | None:
    """Convert a Matplotlib linestyle to Plotly dash style."""
    if linestyle is None:
        return None
    return MPL_TO_PLOTLY_DASH.get(linestyle, linestyle)


def to_plotly_marker(marker: str | None) -> str | None:
    """Convert a Matplotlib marker to a Plotly marker symbol."""
    if marker is None:
        return None
    return MPL_TO_PLOTLY_MARKER.get(marker, marker)


def resolve_legend_visibility(spec: LineSpec | FillSpec | FilledLineSpec, *, allow_legend: bool = True) -> bool:
    """Return whether a visual element should participate in the legend."""
    return allow_legend and spec.get("legend", True)


def resolve_legend_label(spec: LineSpec | FillSpec | FilledLineSpec) -> str | None:
    """Return the effective legend label for a visual element."""
    return spec.get("label")


def resolve_legend_group(spec: LineSpec | FillSpec | FilledLineSpec) -> str | None:
    """Return the effective Plotly legend group for a visual element."""
    return spec.get("legend_group") or resolve_legend_label(spec)


def get_matplotlib_legend_label(
    spec: LineSpec | FillSpec | FilledLineSpec,
    *,
    allow_legend: bool = True,
) -> str | None:
    """Return the label to pass to Matplotlib, or ``None``."""
    if not resolve_legend_visibility(spec, allow_legend=allow_legend):
        return None
    return resolve_legend_label(spec)


def get_plotly_legend_kwargs(
    spec: LineSpec | FillSpec | FilledLineSpec,
    *,
    allow_legend: bool = True,
    primary_trace: bool = True,
) -> dict[str, Any]:
    """Return Plotly legend kwargs for a logical visual element."""
    label = resolve_legend_label(spec)
    legend_group = resolve_legend_group(spec)
    show_legend = resolve_legend_visibility(spec, allow_legend=allow_legend) and primary_trace and label is not None

    kwargs: dict[str, Any] = {"showlegend": show_legend}
    if label is not None:
        kwargs["name"] = label
    if legend_group is not None:
        kwargs["legendgroup"] = legend_group
    return kwargs


def line_spec_to_mpl_kwargs(spec: LineSpec) -> dict[str, Any]:
    """Convert a shared line spec to Matplotlib line kwargs."""
    kwargs: dict[str, Any] = {}
    if "color" in spec:
        kwargs["color"] = spec["color"]
    if "linewidth" in spec:
        kwargs["linewidth"] = spec["linewidth"]
    if "alpha" in spec:
        kwargs["alpha"] = spec["alpha"]
    if "linestyle" in spec:
        kwargs["linestyle"] = translate_linestyle(spec["linestyle"])
    if "marker" in spec:
        kwargs["marker"] = spec["marker"]
    if "markersize" in spec:
        kwargs["markersize"] = spec["markersize"]
    return kwargs


def line_spec_to_plotly_trace_kwargs(spec: LineSpec) -> dict[str, Any]:
    """Convert a shared line spec to Plotly trace kwargs."""
    line: dict[str, Any] = {}
    marker: dict[str, Any] = {}
    kwargs: dict[str, Any] = {}

    color = to_plotly_color(spec.get("color"))
    if color is not None:
        line["color"] = color
        marker["color"] = color
    if "linewidth" in spec:
        line["width"] = spec["linewidth"]
    dash = to_plotly_dash(spec.get("linestyle"))
    if dash is not None:
        line["dash"] = dash
    plotly_marker = to_plotly_marker(spec.get("marker"))
    if plotly_marker is not None:
        marker["symbol"] = plotly_marker
    if "markersize" in spec:
        marker["size"] = spec["markersize"]
    if "alpha" in spec:
        kwargs["opacity"] = spec["alpha"]
    if line:
        kwargs["line"] = line
    if marker:
        kwargs["marker"] = marker
    return kwargs


def fill_spec_to_mpl_kwargs(spec: FillSpec) -> dict[str, Any]:
    """Convert a shared fill spec to Matplotlib fill kwargs."""
    kwargs: dict[str, Any] = {}
    if "fill_color" in spec:
        kwargs["color"] = spec["fill_color"]
    if "fill_alpha" in spec:
        kwargs["alpha"] = spec["fill_alpha"]
    if "edge_color" in spec:
        kwargs["edgecolor"] = spec["edge_color"]
    if "edge_linewidth" in spec:
        kwargs["linewidth"] = spec["edge_linewidth"]
    return kwargs


def fill_spec_to_plotly_trace_kwargs(spec: FillSpec) -> dict[str, Any]:
    """Convert a shared fill spec to Plotly trace kwargs."""
    kwargs: dict[str, Any] = {"fill": "toself"}
    if "fill_color" in spec:
        kwargs["fillcolor"] = to_plotly_color(spec["fill_color"])
    if "fill_alpha" in spec:
        kwargs["opacity"] = spec["fill_alpha"]
    if "edge_linewidth" in spec:
        kwargs["line"] = {"width": spec["edge_linewidth"]}
    return kwargs


def filled_line_spec_to_mpl_line_kwargs(spec: FilledLineSpec) -> dict[str, Any]:
    """Convert a filled line spec to Matplotlib line kwargs."""
    return line_spec_to_mpl_kwargs(spec)


def filled_line_spec_to_mpl_fill_kwargs(spec: FilledLineSpec) -> dict[str, Any]:
    """Convert a filled line spec to Matplotlib fill kwargs."""
    kwargs: dict[str, Any] = {}
    fill_color = spec.get("fill_color", spec.get("color"))
    if fill_color is not None:
        kwargs["color"] = fill_color
    if "fill_alpha" in spec:
        kwargs["alpha"] = spec["fill_alpha"]
    return kwargs


def rgba_color(color: str, alpha: float) -> str:
    """Return an rgba Plotly color string for a line/fill color."""
    plotly_color = to_plotly_color(color) or color
    if plotly_color.startswith("#"):
        r = int(plotly_color[1:3], 16)
        g = int(plotly_color[3:5], 16)
        b = int(plotly_color[5:7], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    if plotly_color.startswith("rgb("):
        return plotly_color.replace("rgb", "rgba").replace(")", f", {alpha})")
    return plotly_color


def filled_line_spec_to_plotly_trace_kwargs(spec: FilledLineSpec) -> dict[str, Any]:
    """Convert a filled line spec to Plotly trace kwargs."""
    kwargs = line_spec_to_plotly_trace_kwargs(spec)
    kwargs["fill"] = "tozeroy"
    fill_alpha = spec.get("fill_alpha")
    fill_color = spec.get("fill_color")
    color = fill_color or spec.get("color")
    if fill_alpha is not None and color is not None:
        kwargs["fillcolor"] = rgba_color(color, fill_alpha)
    elif color is not None:
        kwargs["fillcolor"] = to_plotly_color(color)
    return kwargs


def apply_matplotlib_plot_spec(ax: PltAxes, spec: PlotSpec, *, grid_alpha: float = 0.3) -> None:
    """Apply plot metadata from a shared plot spec to Matplotlib axes."""
    title = spec.get("title")
    if "xlabel" in spec:
        ax.set_xlabel(spec["xlabel"])
    if "ylabel" in spec:
        ax.set_ylabel(spec["ylabel"])
    if spec.get("grid"):
        ax.grid(True, alpha=grid_alpha)
    if title is not None:
        ax.set_title(title)


def apply_matplotlib_twin_plot_spec(ax: PltAxes, spec: PlotSpec) -> None:
    """Apply twin-axis metadata from a shared plot spec to Matplotlib axes."""
    if "ylabel_twin" in spec:
        ax.set_ylabel(spec["ylabel_twin"])


def apply_plotly_plot_spec(
    figure: GoFigure,
    row: int,
    col: int,
    spec: PlotSpec,
    *,
    grid_width: int = 1,
    grid_color: str = "lightgray",
) -> None:
    """Apply plot metadata from a shared plot spec to a Plotly subplot."""
    figure.update_xaxes(
        title_text=spec.get("xlabel"),
        showgrid=spec.get("grid", False),
        gridwidth=grid_width,
        gridcolor=grid_color,
        row=row,
        col=col,
    )
    figure.update_yaxes(
        title_text=spec.get("ylabel"),
        showgrid=spec.get("grid", False),
        gridwidth=grid_width,
        gridcolor=grid_color,
        row=row,
        col=col,
    )
    if spec.get("title") is not None:
        annotation_index = get_plotly_subplot_annotation_index(figure, row, col)
        if len(figure.layout.annotations) > annotation_index:
            figure.layout.annotations[annotation_index].text = spec["title"]


def apply_plotly_twin_plot_spec(figure: GoFigure, row: int, col: int, spec: PlotSpec) -> None:
    """Apply twin-axis metadata from a shared plot spec to a Plotly subplot."""
    figure.update_yaxes(title_text=spec.get("ylabel_twin"), row=row, col=col, secondary_y=True)


def get_subplot_y_limits(fig: GoFigure, axes: GoAxes) -> tuple[float, float]:
    """
    Extract min and max y-values from actual data traces for a Plotly subplot.

    This function analyzes all traces associated with a specific subplot
    and computes the data range, adding a small padding to ensure comfortable
    viewing margins.

    Parameters
    ----------
    fig
        Plotly figure containing the subplot.
    axes
        Subplot position as a tuple (row, column). Row and column indices
        are 1-based as used in Plotly's subplot addressing.

    Returns
    -------
    tuple[float, float]
        (y_min, y_max) representing the recommended y-axis limits for the
        subplot, with padding applied. Returns (0.0, 1.0) if no data found.

    Raises
    ------
    ValueError
        If no finite y values are found in the subplot traces.

    Notes
    -----
    The function uses a hack with dynamic attributes to track whether padding
    has already been applied for a given subplot. Padding is applied only once
    per subplot to avoid recursive expansion.

    The y-axis mapping is determined using Plotly's internal `_grid_ref`
    structure. If not available, a fallback mapping based on row number is used.

    Examples
    --------
    >>> import plotly.graph_objects as go
    >>> fig = go.Figure()
    >>> fig.add_trace(go.Scatter(x=[0,1,2], y=[5,10,15]), row=1, col=1)
    >>> y_min, y_max = get_subplot_y_limits(fig, (1, 1))
    >>> print(f"{y_min:.2f}, {y_max:.2f}")
    4.75, 15.25
    """
    # Extract row and col (1-based indices)
    row, col = axes

    # Determine if padding has already been applied for this subplot
    padding_flag_attr = f"_hack_padding_flag_{row}_{col}"
    padding = 0.00 if hasattr(fig, padding_flag_attr) else 0.05
    if not hasattr(fig, padding_flag_attr):
        setattr(fig, padding_flag_attr, True)

    # Map subplot to yaxis key using _grid_ref
    if hasattr(fig, "_grid_ref") and fig._grid_ref:
        # Find which yaxis this subplot uses
        grid_row = fig._grid_ref[row - 1][col - 1]
        yaxis_key = grid_row[0].layout_keys[1]  # Second element is yaxis key
    else:
        # Fallback mapping: yaxis for first row, yaxis2, yaxis3, etc.
        yaxis_key = "yaxis" if row == 1 else f"yaxis{row}"

    # Collect all y values from traces that use this yaxis
    all_y = []
    for trace in fig.data:
        trace_yaxis = getattr(trace, "yaxis", "y")
        trace_yaxis_key = f"yaxis{trace_yaxis[1:]}" if trace_yaxis != "y" else "yaxis"

        if trace_yaxis_key == yaxis_key and hasattr(trace, "y") and trace.y is not None:
            y_vals = trace.y
            if hasattr(y_vals, "__iter__") and not isinstance(y_vals, str):
                all_y.extend([y for y in y_vals if y is not None])
            elif y_vals is not None:
                all_y.append(y_vals)
    # Filter out 'nan'
    all_y = [y for y in all_y if not np.isnan(y)]
    if all_y:
        y_min, y_max = min(all_y), max(all_y)
        y_range = y_max - y_min
        if y_range == 0:
            y_range = abs(y_min) if y_min != 0 else 1
        y_min = y_min - (y_range * padding)
        y_max = y_max + (y_range * padding)
        return y_min, y_max
    raise ValueError("Could not infer y-axis range because no finite y values were found")
