# -*- coding: ascii -*-

"""
Module for computing the detection delay in online change point detection.

Evaluates how quickly an online algorithm reacts to a true change point.
Missed change points are penalized with the maximum allowable delay.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.abstracts.isingle_run_metric import ISingleRunMetric
from pysatl_cpd.analysis.metrics.single_run.utils import match_change_points
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData as LabeledData


class Delays[TraceT: OnlineDetectionTrace, ProviderT: LabeledData](ISingleRunMetric[TraceT, ProviderT, list[int]]):
    """
    Computes the detection delay for online change point detection algorithms.

    Delay is defined as the distance between an actual change point and its
    corresponding detection. If a change point is missed (False Negative),
    the algorithm is penalized with the maximum defined delay.

    Parameters
    ----------
    max_delay
        The maximum allowable delay window for a valid detection. Also used
        as the penalty value for missed detections (FN).

    Raises
    ------
    ValueError
        If maximum delay is negative
    """

    def __init__(self, max_delay: int) -> None:
        if max_delay < 0:
            raise ValueError("Maximum delay must be non-negative")

        self._max_delay = max_delay

    def evaluate(self, run: SingleRun[TraceT, ProviderT]) -> list[int]:
        """
        Calculate delays for all true change points

        Delay is computed per true change point as the minimum non-negative delay
        among matched detections within [true_change, true_change + max_delay].
        If there is no match, `max_delay` is returned for that change point.

        Parameters
        ----------
        run
            The run to evaluate.

        Returns
        -------
        list[int]
            A list of delays where each element corresponds to the true
            change point at the same index in ``data.change_points``.
            Length is exactly equal to the number of true change points.
        """
        matching = match_change_points(
            run.trace.detected_change_points, run.provider.change_points, (0, self._max_delay)
        )
        return [min(detections) - cp if detections else self._max_delay for cp, detections in matching.items()]
