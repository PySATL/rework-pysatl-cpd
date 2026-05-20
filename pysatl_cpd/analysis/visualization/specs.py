# -*- coding: ascii -*-
"""
Shared visualization style specifications.

This module defines backend-agnostic visual-only TypedDict specifications used
across visualization components and visualizers.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import TypedDict


class PlotSpec(TypedDict, total=False):
    """Visual metadata for a subplot."""

    title: str | None
    xlabel: str
    ylabel: str
    ylabel_twin: str
    grid: bool


class LineSpec(TypedDict, total=False):
    """Visual style for a line-like element."""

    label: str | None
    legend: bool
    legend_group: str | None
    color: str
    linewidth: float
    alpha: float
    linestyle: str
    marker: str
    markersize: float


class FilledLineSpec(TypedDict, total=False):
    """Visual style for a line with optional area fill."""

    label: str | None
    legend: bool
    legend_group: str | None
    color: str
    linewidth: float
    alpha: float
    linestyle: str
    marker: str
    markersize: float
    fill_color: str
    fill_alpha: float


class FillSpec(TypedDict, total=False):
    """Visual style for a filled element."""

    label: str | None
    legend: bool
    legend_group: str | None
    fill_color: str
    fill_alpha: float
    edge_color: str
    edge_linewidth: float
