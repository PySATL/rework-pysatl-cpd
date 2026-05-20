# -*- coding: ascii -*-
"""Compatibility layer for the plain multivariate visualizer."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.visualization.timeseries.plain_multivariate_timeseries_visualizer import (
    PlainMultivariateTimeseriesVisualizer,
)

MultivariateTimeseriesVisualizer = PlainMultivariateTimeseriesVisualizer

__all__ = ["MultivariateTimeseriesVisualizer"]
