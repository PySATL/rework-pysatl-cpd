# -*- coding: ascii -*-

"""
Base module defining the interface for multi-run evaluation metrics.

This module provides the generic `IMultipleRunMetric` source class, establishing
the standard evaluation protocol for change point detection algorithms
over a series of labeled data providers.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData as LabeledData


class IMultipleRunMetric[TraceT: DetectionTrace, ProviderT: LabeledData[Any, Any], ResultT](ABC):
    """
    Abstract source class for all multi-run evaluation metrics.

    Provides a generic interface to evaluate a collection of detection traces
    against their corresponding labeled ground truth datasets.
    """

    @abstractmethod
    def evaluate(self, runs: Sequence[SingleRun[TraceT, ProviderT]]) -> ResultT:  # pragma: no cover
        """
        Evaluate the detection traces against the provided labeled data.

        Parameters
        ----------
        runs
            A sequence of runs containing a detection trace and the
            corresponding labeled data provider.

        Returns
        -------
        ResultT
            The computed metric result over the entire dataset.
        """
        ...
