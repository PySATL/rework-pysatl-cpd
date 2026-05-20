# -*- coding: ascii -*-
"""Compatibility helpers for metric matching."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.single_run.utils import match_change_points
from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData


def match[TraceT: DetectionTrace, ProviderT: LabeledData](
    run: SingleRun[TraceT, ProviderT], error_margin: tuple[int, int]
) -> dict[int, set[int]]:
    """Match detections to true change points for one run.

    Parameters
    ----------
    run
        The single run to evaluate.
    error_margin
        (left, right) tolerance around each true change point.

    Returns
    -------
    dict[int, set[int]]
    """
    return match_change_points(run.trace.detected_change_points, run.provider.change_points, error_margin)
