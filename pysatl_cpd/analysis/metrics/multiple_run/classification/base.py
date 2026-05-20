# -*- coding: ascii -*-

"""Total true positive metrics over multiple runs."""  # TODO: update docstring

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Any

from pysatl_cpd.analysis.metrics.multiple_run.aggregation_metric import TotalSum
from pysatl_cpd.analysis.metrics.single_run import (
    FalseNegativeCount as SingleFN,
    FalsePositiveCount as SingleFP,
    TruePositiveCount as SingleTP,
)
from pysatl_cpd.core import DetectionTrace
from pysatl_cpd.data import LabeledData as LabeledData


class TotalFP[TraceT: DetectionTrace, ProviderT: LabeledData[Any, Any]](TotalSum[TraceT, ProviderT, int]):
    """Configure the total false positive metric with an error margin.

    Parameters
    ----------
    error_margin
        Allowed (left, right) margin around each true change point for
        matching detections.
    """

    def __init__(self, error_margin: tuple[int, int]) -> None:
        self._base_metric = SingleFP[TraceT, ProviderT](error_margin)

    @property
    def base_metric(self) -> SingleFP[TraceT, ProviderT]:
        """Per-run false-positive metric.

        Returns
        -------
        SingleFP
        """
        return self._base_metric


class TotalFN[TraceT: DetectionTrace, ProviderT: LabeledData[Any, Any]](TotalSum[TraceT, ProviderT, int]):
    """Configure the total false negative metric with an error margin.

    Parameters
    ----------
    error_margin
        Allowed (left, right) margin around each true change point for
        matching detections.
    """

    def __init__(self, error_margin: tuple[int, int]) -> None:
        self._base_metric = SingleFN[TraceT, ProviderT](error_margin)

    @property
    def base_metric(self) -> SingleFN[TraceT, ProviderT]:
        """Per-run false-negative metric.

        Returns
        -------
        SingleFN
        """
        return self._base_metric


class TotalTP[TraceT: DetectionTrace, ProviderT: LabeledData[Any, Any]](TotalSum[TraceT, ProviderT, int]):
    """Configure the total true positive metric with an error margin.

    Parameters
    ----------
    error_margin
        Allowed (left, right) margin around each true change point for
        matching detections.
    """

    def __init__(self, error_margin: tuple[int, int]) -> None:
        self._base_metric = SingleTP[TraceT, ProviderT](error_margin)

    @property
    def base_metric(self) -> SingleTP[TraceT, ProviderT]:
        """Per-run true-positive metric.

        Returns
        -------
        SingleTP
        """
        return self._base_metric
