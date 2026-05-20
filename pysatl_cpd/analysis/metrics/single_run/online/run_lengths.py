# -*- coding: ascii -*-

"""
Module for computing Run Lengths between consecutive alarms (detections)
for online algorithms.

In this implementation run length is simply the distance (in time steps)
between consecutive detected change points ('positives'). The first run length
is measured from time step 0 to the first detection.

Note
----
Ground truth (`data`) is not used here. Metrics that require TP/FP
separation should be implemented separately (e.g., via classification metrics).
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import cast

import numpy as np

from pysatl_cpd.analysis.metrics.abstracts.isingle_run_metric import ISingleRunMetric
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData as LabeledData


class RunLengths[TraceT: OnlineDetectionTrace, ProviderT: LabeledData](ISingleRunMetric[TraceT, ProviderT, list[int]]):
    """
    Computes run lengths between consecutive detections.

    Run length is the distance between two successive detected change points,
    starting from time 0. This is the definition used for Average Run Length
    (ARL) - every detection is treated as a positive, ground-truth is ignored.
    """

    def evaluate(self, run: SingleRun[TraceT, ProviderT]) -> list[int]:
        """
        Calculate run lengths between consecutive detections.

        Parameters
        ----------
        run
            The run to evaluate. Only `run.trace` is used.

        Returns
        -------
        list[int]
            Distances between consecutive detections, with the first distance measured
            from 0 to the first detection.
        """

        detected_changes = run.trace.detected_change_points
        if not detected_changes:
            return [len(run.provider)]

        run_lengths = np.diff(np.insert(np.sort(detected_changes), 0, 0))
        return cast(list[int], run_lengths.tolist())
