# -*- coding: ascii -*-
"""Shared scaffolding for multivariate time-series visualizers."""

from __future__ import annotations

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC
from collections.abc import Sequence
from typing import Any, Self, cast

import pandas as pd

from pysatl_cpd.analysis.visualization.abstracts import ITimeseriesVisualizer
from pysatl_cpd.analysis.visualization.specs import LineSpec, PlotSpec
from pysatl_cpd.analysis.visualization.typedefs import DrawBackend
from pysatl_cpd.data import DataProvider
from pysatl_cpd.data.typedefs import TimeseriesAnnotation

type AxisSelector = int | str


class AbstractMultivariateTimeseriesVisualizer[DataProviderT: DataProvider[Any, TimeseriesAnnotation]](
    ITimeseriesVisualizer[DataProviderT],
    ABC,
):
    """Common provider and style management for multivariate visualizers."""

    def __init__(self, backend: DrawBackend | str, *, dimensionality: int | None = None) -> None:
        super().__init__(backend)
        if dimensionality is not None and dimensionality <= 0:
            raise ValueError("dimensionality must be positive")

        self._dimensionality = dimensionality
        self._data_provider: DataProviderT | None = None
        self._time_column: str | None = None

    def set_data_provider(self, data_provider: DataProviderT) -> Self:
        """Store a validated provider for subsequent drawing."""
        self._validate_provider(data_provider)
        self._validate_time_column(data_provider, self._time_column)
        self._data_provider = data_provider
        return self

    def set_time_column(self, time_column: str | None) -> Self:
        """Store the optional x-axis source column name."""
        self._validate_time_column(self._data_provider, time_column)
        self._time_column = time_column
        return self

    def _validate_provider(self, data_provider: DataProviderT) -> None:
        """Validate provider compatibility for the concrete visualizer."""

    def _validate_time_column(
        self,
        data_provider: DataProviderT | None,
        time_column: str | None,
    ) -> None:
        """Validate the optional time column for the concrete visualizer."""

    def _resolve_axis_indices(self, axes: Sequence[AxisSelector]) -> list[int]:
        """Resolve integer or named axis selectors to dimension indices."""
        if self._dimensionality is None:
            raise ValueError("Axis selectors require a fixed dimensionality")
        if not axes:
            raise ValueError("axes must contain at least one axis selector")

        resolved: list[int] = []
        provider_columns = self._get_provider_columns(self._data_provider)
        for axis in axes:
            if isinstance(axis, int):
                if axis < 0 or axis >= self._dimensionality:
                    raise ValueError(f"Axis index {axis} is out of bounds for dimensionality {self._dimensionality}")
                resolved.append(axis)
                continue

            if not isinstance(axis, str):
                raise TypeError(f"Unsupported axis selector type: {type(axis).__name__}")
            if provider_columns is None:
                raise ValueError("Named axes require a data provider with accessible columns")
            if axis not in provider_columns:
                raise ValueError(f"Unknown axis name '{axis}'. Available columns: {provider_columns}")
            resolved.append(provider_columns.index(axis))

        return resolved

    def _get_provider_columns(self, data_provider: DataProviderT | None) -> list[str] | None:
        """Return feature column names when the provider exposes them."""
        if data_provider is None:
            return None
        if hasattr(data_provider, "columns"):
            return list(cast(Any, data_provider).columns)
        if hasattr(data_provider, "feature_columns"):
            return list(cast(Any, data_provider).feature_columns)

        unlabeled = getattr(data_provider, "unlabeled", None)
        if unlabeled is not None and hasattr(unlabeled, "columns"):
            return list(cast(Any, unlabeled).columns)
        return None

    def _get_provider_dataset(self, data_provider: DataProviderT) -> pd.DataFrame | None:
        """Return a pandas dataset when the provider exposes one."""
        if hasattr(data_provider, "dataset"):
            dataset = cast(Any, data_provider).dataset
            return cast(pd.DataFrame, dataset() if callable(dataset) else dataset)

        unlabeled = getattr(data_provider, "unlabeled", None)
        if unlabeled is not None and hasattr(unlabeled, "dataset"):
            dataset = cast(Any, unlabeled).dataset
            return cast(pd.DataFrame, dataset() if callable(dataset) else dataset)
        return None

    def _resolve_time_points(self, data_provider: DataProviderT, length: int) -> list[Any]:
        """Resolve x-axis values from the configured time column or indices."""
        dataset = self._get_provider_dataset(data_provider)
        if self._time_column is not None and dataset is not None:
            return cast(list[Any], dataset[self._time_column].tolist())
        return list(range(length))

    @staticmethod
    def _make_default_plot_opts(*, ylabel: str = "Value") -> PlotSpec:
        """Build a default visual-only subplot spec."""
        return {
            "xlabel": "Time Index",
            "ylabel": ylabel,
            "grid": True,
            "title": None,
        }

    @staticmethod
    def _make_default_line_opts(label: str | None, *, color: str | None = None) -> LineSpec:
        """Build a default visual-only line spec."""
        spec: LineSpec = {
            "linewidth": 1.5,
            "alpha": 1.0,
            "label": label,
            "legend": True,
        }
        if color is not None:
            spec["color"] = color
        return spec
