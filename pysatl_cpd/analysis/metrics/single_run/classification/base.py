# -*- coding: ascii -*-

"""Base classes for single-run classification metrics."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import abstractmethod
from collections.abc import Sequence

from pysatl_cpd.analysis.metrics.abstracts.isingle_run_metric import ISingleRunMetric
from pysatl_cpd.analysis.metrics.single_run.utils import match_change_points, validate_error_margin
from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData as LabeledData


class ClassificationMetricBase[TraceT: DetectionTrace, ProviderT: LabeledData, ResultT](
    ISingleRunMetric[TraceT, ProviderT, ResultT]
):
    """Base class for single-run metrics built on change-point matching.

    Parameters
    ----------
    error_margin
        Allowed (left, right) margin for matching detections to true CPs.
    """

    def __init__(self, error_margin: tuple[int, int]) -> None:
        self._error_margin = validate_error_margin(error_margin)

    def match(
        self,
        detected_change_points: Sequence[int],
        true_change_points: Sequence[int],
    ) -> dict[int, set[int]]:
        """Match detections to true change points using stored margin.

        Parameters
        ----------
        detected_change_points
            Detected change-point indices.
        true_change_points
            Ground-truth change-point indices.

        Returns
        -------
        dict[int, set[int]]
        """
        return match_change_points(detected_change_points, true_change_points, self._error_margin)

    @abstractmethod
    def evaluate(self, run: SingleRun[TraceT, ProviderT]) -> ResultT:  # pragma: no cover
        """Evaluate the metric for a single run.

        Parameters
        ----------
        run
            The run to evaluate.

        Returns
        -------
        ResultT
        """
        ...


class ClassificationPrimitive[TraceT: DetectionTrace, ProviderT: LabeledData](
    ClassificationMetricBase[TraceT, ProviderT, int]
):
    """Base class for count-based single-run classification metrics."""

    @abstractmethod
    def evaluate(self, run: SingleRun[TraceT, ProviderT]) -> int:  # pragma: no cover
        """Evaluate the metric for a single run.

        Parameters
        ----------
        run
            The run to evaluate.

        Returns
        -------
        int
        """
        ...
