# -*- coding: ascii -*-

"""Single-run false negative count metric."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.single_run.classification.base import ClassificationPrimitive
from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData as LabeledData


class FalseNegativeCount[TraceT: DetectionTrace, ProviderT: LabeledData](ClassificationPrimitive[TraceT, ProviderT]):
    """Count true change points that have no matched detection."""

    def evaluate(self, run: SingleRun[TraceT, ProviderT]) -> int:
        """Count true change points that have no matched detection.

        Parameters
        ----------
        run
            A single detection run with trace and labeled provider.

        Returns
        -------
        int
            The false negative count.
        """
        matching = self.match(run.trace.detected_change_points, run.provider.change_points)
        return sum(not matches for matches in matching.values())
