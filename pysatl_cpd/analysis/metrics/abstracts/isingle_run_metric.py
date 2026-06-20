# -*- coding: ascii -*-

"""Base interface for metrics evaluated on one `SingleRun`."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData as LabeledData


class ISingleRunMetric[TraceT: DetectionTrace, ProviderT: LabeledData, ResultT](ABC):
    """Base class for metrics that evaluate one run.

    Notes
    -----
    The generic parameters identify the detection trace type, labeled data
    provider type, and metric result type.
    """

    @abstractmethod
    def evaluate(self, run: SingleRun[TraceT, ProviderT]) -> ResultT:  # pragma: no cover
        """Evaluate one trace/provider pair.

        Parameters
        ----------
        run
            The run to evaluate.

        Returns
        -------
        ResultT
        """
        ...
