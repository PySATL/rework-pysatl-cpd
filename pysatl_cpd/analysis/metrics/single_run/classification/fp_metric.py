# -*- coding: ascii -*-

"""Single-run false positive count metric."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.single_run.classification.base import ClassificationPrimitive
from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData as LabeledData


class FalsePositiveCount[TraceT: DetectionTrace, ProviderT: LabeledData](ClassificationPrimitive[TraceT, ProviderT]):
    """Count detections that are not matched to any true change point."""

    def evaluate(self, run: SingleRun[TraceT, ProviderT]) -> int:
        """Count detections that are not matched to any true change point.

        Parameters
        ----------
        run
            A single detection run with trace and labeled provider.

        Returns
        -------
        int
            The false positive count.
        """
        matching = self.match(run.trace.detected_change_points, run.provider.change_points)
        return len(run.trace.detected_change_points) - sum(len(matches) for matches in matching.values())
