# -*- coding: ascii -*-
"""
Time series visualizer interface.

This module defines the abstract source class for visualizers that render
time series data with change point annotations.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod
from typing import Any, Self

from pysatl_cpd.analysis.visualization.abstracts.ivisualizer import IVisualizer
from pysatl_cpd.data import DataProvider
from pysatl_cpd.data.typedefs import TimeseriesAnnotation


class ITimeseriesVisualizer[DataProviderT: DataProvider[Any, TimeseriesAnnotation]](IVisualizer, ABC):
    """
    Abstract source class for time series visualizers.

    Visualizers of this type render the original time series data,
    optionally with ground truth change points, detected change points,
    and annotation of learning and skip periods.

    Notes
    -----
    The generic DataProviderT type is bound by DataProvider and contains the
    observations to be visualized. The bound ensures that any concrete
    implementation works with valid data providers.
    """

    @abstractmethod
    def set_data_provider(self, data_provider: DataProviderT) -> Self:  # pragma: no cover
        """
        Set the data provider containing the time series observations.

        Parameters
        ----------
        data_provider
            Data provider that yields observations sequentially.

        Returns
        -------
        Self
            Returns self to allow method chaining.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError
