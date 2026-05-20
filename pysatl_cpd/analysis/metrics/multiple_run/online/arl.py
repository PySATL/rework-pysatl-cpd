# -*- coding: ascii -*-

"""Average run length over multiple runs."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

from pysatl_cpd.analysis.metrics.multiple_run.aggregation_metric import TotalMean
from pysatl_cpd.analysis.metrics.single_run.online.run_lengths import RunLengths
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData as LabeledData


class ARLMetric[TraceT: OnlineDetectionTrace[Any], ProviderT: LabeledData[Any, Any]](TotalMean[TraceT, ProviderT, int]):
    """Configure the ARL metric."""

    def __init__(self) -> None:
        self._base_metric = RunLengths[TraceT, ProviderT]()

    @property
    def base_metric(self) -> RunLengths[TraceT, ProviderT]:
        """Per-run run-length metric.

        Returns
        -------
        RunLengths
        """
        return self._base_metric
