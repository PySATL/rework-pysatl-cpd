# -*- coding: ascii -*-

"""Base class for multi-run metrics derived from other multi-run metrics."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from pysatl_cpd.analysis.metrics.abstracts.imultiple_run_metric import IMultipleRunMetric
from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData as LabeledData


class DerivedMetric[TraceT: DetectionTrace, ProviderT: LabeledData[Any, Any], ResultInT, ResultOutT](
    IMultipleRunMetric[TraceT, ProviderT, ResultOutT]
):
    """Evaluate several multi-run metrics and combine their outputs.

    Notes
    -----
    The generic parameters identify the detection trace type, labeled data
    provider type, input metric result type, and derived metric result type.
    """

    @property
    @abstractmethod
    def bases(self) -> Mapping[str, IMultipleRunMetric[TraceT, ProviderT, ResultInT]]:  # pragma: no cover
        """Underlying metrics used to compute the derived value.

        Returns
        -------
        Mapping[str, IMultipleRunMetric]
        """
        ...

    @abstractmethod
    def compute(self, values: Mapping[str, ResultInT]) -> ResultOutT:  # pragma: no cover
        """Combine already-aggregated metric values into the final result.

        Parameters
        ----------
        values
            Named aggregated metric values.

        Returns
        -------
        ResultOutT
        """
        ...

    def evaluate(self, runs: Sequence[SingleRun[TraceT, ProviderT]]) -> ResultOutT:
        """Evaluate all source metrics on each run and compute the derived result.

        Parameters
        ----------
        runs
            Sequence of single runs to evaluate.

        Returns
        -------
        ResultOutT
        """
        bases_values = {name: metric.evaluate(runs) for name, metric in self.bases.items()}
        return self.compute(bases_values)
