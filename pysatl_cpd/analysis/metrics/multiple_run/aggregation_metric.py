# -*- coding: ascii -*-

"""Base classes for reducers over single-run metric results."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import abstractmethod
from collections.abc import Sequence
from statistics import fmean, median
from typing import Any, cast, overload

from pysatl_cpd.analysis.metrics.abstracts.imultiple_run_metric import IMultipleRunMetric
from pysatl_cpd.analysis.metrics.abstracts.isingle_run_metric import ISingleRunMetric
from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData as LabeledData
from pysatl_cpd.typedefs import Number


class AggregationMetric[TraceT: DetectionTrace, ProviderT: LabeledData[Any, Any], ResultInT, ResultOutT](
    IMultipleRunMetric[TraceT, ProviderT, ResultOutT]
):
    """Evaluate a single-run metric on each run and aggregate the results.

    Type Parameters
    ---------------
    TraceT
        Detection trace type.
    ProviderT
        Labeled data provider type.
    ResultInT
        Per-run metric result type.
    ResultOutT
        Aggregated result type.
    """

    @property
    @abstractmethod
    def base_metric(self) -> ISingleRunMetric[TraceT, ProviderT, ResultInT]:  # pragma: no cover
        """Underlying per-run metric.

        Returns
        -------
        ISingleRunMetric
        """

    @abstractmethod
    def aggregate(self, results: Sequence[ResultInT]) -> ResultOutT:  # pragma: no cover
        """Aggregate a sequence of single-run metric results into a final value.

        Parameters
        ----------
        results
            Per-run metric results.

        Returns
        -------
        ResultOutT
        """
        ...

    def evaluate(self, runs: Sequence[SingleRun[TraceT, ProviderT]]) -> ResultOutT:
        """Evaluate the source metric on each run and aggregate.

        Parameters
        ----------
        runs
            Sequence of single runs to evaluate.

        Returns
        -------
        ResultOutT
        """
        results = [self.base_metric.evaluate(run) for run in runs]
        return self.aggregate(results)


def _normalize_numeric_results[NumberT: Number](
    results: Sequence[NumberT | Sequence[NumberT]],
) -> list[NumberT]:
    """Normalise flat or nested numeric results into one flat list.

    Parameters
    ----------
    results
        Sequence of numbers or sequences of numbers.

    Returns
    -------
    list[NumberT]
        Flattened list.
    """
    if not results:
        return []

    normalized: list[NumberT] = []
    for result in results:
        if isinstance(result, Sequence):
            normalized.extend(result)
        else:
            normalized.append(result)
    return normalized


class TotalSum[TraceT: DetectionTrace, ProviderT: LabeledData[Any, Any], NumberT: Number](
    AggregationMetric[TraceT, ProviderT, NumberT | Sequence[NumberT], NumberT]
):
    """Sum flat or nested per-run numeric results into one total."""

    @overload
    def aggregate(self, results: Sequence[NumberT]) -> NumberT: ...  # pragma: no cover

    @overload
    def aggregate(self, results: Sequence[Sequence[NumberT]]) -> NumberT: ...  # pragma: no cover

    @overload
    def aggregate(self, results: Sequence[NumberT | Sequence[NumberT]]) -> NumberT: ...  # pragma: no cover

    def aggregate(self, results: Sequence[NumberT | Sequence[NumberT]]) -> NumberT:
        """Sum all numeric results across runs.

        Parameters
        ----------
        results
            Per-run metric results, possibly nested.

        Returns
        -------
        NumberT
            The total sum.
        """
        return cast(NumberT, sum(_normalize_numeric_results(results)))


class TotalMean[TraceT: DetectionTrace, ProviderT: LabeledData[Any, Any], NumberT: Number](
    AggregationMetric[TraceT, ProviderT, NumberT | Sequence[NumberT], float]
):
    """Take the global mean of flat or nested per-run numeric results."""

    @property
    def _value_on_empty(self) -> float | None:
        """Return the fallback value for empty inputs, or ``None`` to raise."""
        return None

    @overload
    def aggregate(self, results: Sequence[NumberT]) -> float: ...  # pragma: no cover

    @overload
    def aggregate(self, results: Sequence[Sequence[NumberT]]) -> float: ...  # pragma: no cover

    @overload
    def aggregate(self, results: Sequence[NumberT | Sequence[NumberT]]) -> float: ...  # pragma: no cover

    def aggregate(self, results: Sequence[NumberT | Sequence[NumberT]]) -> float:
        """Compute the mean of all numeric results across runs.

        Parameters
        ----------
        results
            Per-run metric results, possibly nested.

        Returns
        -------
        float
            The arithmetic mean.

        Raises
        ------
        ValueError
            If the results sequence is empty.
        """
        normalized = _normalize_numeric_results(results)
        if not normalized:
            if self._value_on_empty is None:
                raise ValueError("Cannot aggregate mean of an empty sequence")
            return self._value_on_empty
        return float(fmean(normalized))


class TotalMedian[TraceT: DetectionTrace, ProviderT: LabeledData[Any, Any], NumberT: Number](
    AggregationMetric[TraceT, ProviderT, NumberT | Sequence[NumberT], float]
):
    """Take the global median of flat or nested per-run numeric results."""

    @property
    def _value_on_empty(self) -> float | None:
        """Return the fallback value for empty inputs, or ``None`` to raise."""
        return None

    @overload
    def aggregate(self, results: Sequence[NumberT]) -> float: ...  # pragma: no cover

    @overload
    def aggregate(self, results: Sequence[Sequence[NumberT]]) -> float: ...  # pragma: no cover

    @overload
    def aggregate(self, results: Sequence[NumberT | Sequence[NumberT]]) -> float: ...  # pragma: no cover

    def aggregate(self, results: Sequence[NumberT | Sequence[NumberT]]) -> float:
        """Compute the median of all numeric results across runs.

        Parameters
        ----------
        results
            Per-run metric results, possibly nested.

        Returns
        -------
        float
            The median value.

        Raises
        ------
        ValueError
            If the results sequence is empty.
        """
        normalized = _normalize_numeric_results(results)
        if not normalized:
            if self._value_on_empty is None:
                raise ValueError("Cannot aggregate median of an empty sequence")
            return self._value_on_empty
        return float(median(normalized))
