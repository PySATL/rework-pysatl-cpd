# -*- coding: ascii -*-

"""F-beta score over multiple runs."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from typing import Any

from pysatl_cpd.analysis.metrics.abstracts.imultiple_run_metric import IMultipleRunMetric
from pysatl_cpd.analysis.metrics.multiple_run.classification.precision import PrecisionMetric
from pysatl_cpd.analysis.metrics.multiple_run.classification.recall import RecallMetric
from pysatl_cpd.analysis.metrics.multiple_run.derived_metric import DerivedMetric
from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData as LabeledData


class FScoreMetric[TraceT: DetectionTrace, ProviderT: LabeledData[Any, Any]](
    DerivedMetric[TraceT, ProviderT, float, float]
):
    """Configure the F-beta metric with an error margin.

    Parameters
    ----------
    error_margin
        Allowed (left, right) margin around each true change point for
        matching detections.
    beta
        Weight of recall relative to precision. ``beta=1`` gives the F1
        score, larger values emphasize recall, and smaller values
        emphasize precision.

    Raises
    ------
    ValueError
        If ``beta`` is negative.
    """

    def __init__(self, error_margin: tuple[int, int], beta: float = 1.0) -> None:
        if beta < 0:
            raise ValueError("beta must be non-negative")

        self._beta = float(beta)
        self._bases = {
            "precision": PrecisionMetric[TraceT, ProviderT](error_margin),
            "recall": RecallMetric[TraceT, ProviderT](error_margin),
        }

    @property
    def bases(self) -> Mapping[str, IMultipleRunMetric[TraceT, ProviderT, float]]:
        """Underlying precision and recall metrics.

        Returns
        -------
        Mapping[str, IMultipleRunMetric]
        """
        return self._bases

    def compute(self, values: Mapping[str, float]) -> float:
        """Compute the F-beta score from precision and recall.

        Parameters
        ----------
        values
            Must contain ``precision`` and ``recall`` keys.

        Returns
        -------
        float
            The F-beta score. Returns 0.0 when the weighted denominator is
            zero.
        """
        precision = values["precision"]
        recall = values["recall"]
        beta_squared = self._beta**2
        denominator = beta_squared * precision + recall
        if denominator == 0:
            return 0.0

        return float((1.0 + beta_squared) * precision * recall / denominator)
