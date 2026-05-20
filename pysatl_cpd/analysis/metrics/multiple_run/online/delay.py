# -*- coding: ascii -*-

"""Aggregated detection delays over multiple runs."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

from pysatl_cpd.analysis.metrics.multiple_run.aggregation_metric import TotalMean, TotalMedian
from pysatl_cpd.analysis.metrics.single_run.online.delays import Delays
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData as LabeledData


class MeanDelayMetric[TraceT: OnlineDetectionTrace[Any], ProviderT: LabeledData[Any, Any]](
    TotalMean[TraceT, ProviderT, int]
):
    """Configure the mean delay metric with a maximum delay cap.

    Parameters
    ----------
    max_delay
        Maximum allowed delay. Delays exceeding this value are capped.
        Must be non-negative.

    Raises
    ------
    ValueError
        If ``max_delay`` is negative.
    """

    def __init__(self, max_delay: int) -> None:
        if max_delay < 0:
            raise ValueError("Maximum delay must be non-negative")

        self._base_metric = Delays[TraceT, ProviderT](max_delay)
        self._max_delay = max_delay

    @property
    def base_metric(self) -> Delays[TraceT, ProviderT]:
        """Per-run delay metric.

        Returns
        -------
        Delays
        """
        return self._base_metric

    @property
    def _value_on_empty(self) -> float:
        """Fallback value for empty sequences."""
        return float(self._max_delay)


class MedianDelayMetric[TraceT: OnlineDetectionTrace[Any], ProviderT: LabeledData[Any, Any]](
    TotalMedian[TraceT, ProviderT, int]
):
    """Configure the median delay metric with a maximum delay cap.

    Parameters
    ----------
    max_delay
        Maximum allowed delay. Delays exceeding this value are capped.
        Must be non-negative.
    """

    def __init__(self, max_delay: int) -> None:
        self._base_metric = Delays[TraceT, ProviderT](max_delay)
        self._max_delay = max_delay

    @property
    def base_metric(self) -> Delays[TraceT, ProviderT]:
        """Per-run delay metric.

        Returns
        -------
        Delays
        """
        return self._base_metric

    @property
    def _value_on_empty(self) -> float:
        """Fallback value for empty sequences."""
        return float(self._max_delay)
