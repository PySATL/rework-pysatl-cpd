# -*- coding: ascii -*-

"""Micro-averaged precision over multiple runs."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from typing import Any

from pysatl_cpd.analysis.metrics.abstracts.imultiple_run_metric import IMultipleRunMetric
from pysatl_cpd.analysis.metrics.multiple_run.classification.base import TotalFP, TotalTP
from pysatl_cpd.analysis.metrics.multiple_run.derived_metric import DerivedMetric
from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData as LabeledData
from pysatl_cpd.typedefs import Number


class PrecisionMetric[TraceT: DetectionTrace, ProviderT: LabeledData[Any, Any]](
    DerivedMetric[TraceT, ProviderT, Number, float]
):
    """Configure the precision metric with an error margin.

    Parameters
    ----------
    error_margin
        Allowed (left, right) margin around each true change point for
        matching detections.
    """

    def __init__(self, error_margin: tuple[int, int]) -> None:
        self._bases = {
            "tp": TotalTP[TraceT, ProviderT](error_margin),
            "fp": TotalFP[TraceT, ProviderT](error_margin),
        }

    @property
    def bases(self) -> Mapping[str, IMultipleRunMetric[TraceT, ProviderT, int]]:
        """Underlying TP and FP metrics.

        Returns
        -------
        Mapping[str, IMultipleRunMetric]
        """
        return self._bases

    def compute(self, values: Mapping[str, Number]) -> float:
        """Compute precision as TP / (TP + FP).

        Parameters
        ----------
        values
            Must contain ``tp`` and ``fp`` keys.

        Returns
        -------
        float
            The precision score. Returns 1.0 when TP + FP is zero.
        """
        tp = values["tp"]
        fp = values["fp"]
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
