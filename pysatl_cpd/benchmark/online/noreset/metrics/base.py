# -*- coding: ascii -*-
"""Generic no-reset metric wrappers."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from pysatl_cpd.analysis.metrics.abstracts.imultiple_run_metric import IMultipleRunMetric
from pysatl_cpd.analysis.metrics.abstracts.isingle_run_metric import ISingleRunMetric
from pysatl_cpd.analysis.metrics.multiple_run.derived_metric import DerivedMetric
from pysatl_cpd.benchmark.online.noreset.detector.noreset_trace import NoResetDetectionTrace
from pysatl_cpd.benchmark.online.noreset.metrics.policy import NoResetPolicy
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.providers.labeled import LabeledData


class NoResetThresholdMetric[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any], ResultT](Protocol):
    """Protocol for no-reset metrics that evaluate threshold callables over runs."""

    def evaluate(  # pragma: no cover
        self,
        runs: Sequence[SingleRun[OnlineDetectionTrace[StateT], ProviderT]],
    ) -> Callable[[float], ResultT]:
        """Return a threshold-indexed evaluator.

        Parameters
        ----------
        runs
            Sequence of runs to evaluate.

        Returns
        -------
        Callable[[float], ResultT]
        """
        ...


class NoResetSingleRunMetric[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any], ResultT]:
    """Wrap a classical single-run metric with a no-reset policy.

    Parameters
    ----------
    source
        Metric that operates on ``NoResetDetectionTrace``.
    policy
        Policy that transforms raw detection traces into classified traces.
    """

    def __init__(
        self,
        source: ISingleRunMetric[NoResetDetectionTrace[StateT], ProviderT, ResultT],
        policy: NoResetPolicy[StateT, ProviderT],
    ) -> None:
        self.source = source
        self.policy = policy

    def evaluate(
        self,
        run: SingleRun[OnlineDetectionTrace[StateT], ProviderT],
    ) -> Callable[[float], ResultT]:
        """Return a callable that evaluates the wrapped metric at a threshold.

        Parameters
        ----------
        run
            Single run with online detection trace.

        Returns
        -------
        Callable[[float], ResultT]
        """

        def evaluator(threshold: float) -> ResultT:
            """Evaluate the metric at a specific threshold.

            Parameters
            ----------
            threshold
                Detection threshold.

            Returns
            -------
            ResultT
            """
            transformed_run = self.policy.apply(run, threshold)
            return self.source.evaluate(transformed_run)

        return evaluator


class NoResetMultipleRunMetric[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any], ResultT]:
    """Wrap a classical multiple-run metric with a no-reset policy.

    Parameters
    ----------
    source
        Metric that operates on ``NoResetDetectionTrace``.
    policy
        Policy that transforms raw detection traces into classified traces.
    """

    def __init__(
        self,
        source: IMultipleRunMetric[NoResetDetectionTrace[StateT], ProviderT, ResultT],
        policy: NoResetPolicy[StateT, ProviderT],
    ) -> None:
        self.source = source
        self.policy = policy

    def evaluate(
        self,
        runs: Sequence[SingleRun[OnlineDetectionTrace[StateT], ProviderT]],
    ) -> Callable[[float], ResultT]:
        """Return a callable that evaluates the wrapped metric at a threshold.

        Parameters
        ----------
        runs
            Sequence of runs to evaluate.

        Returns
        -------
        Callable[[float], ResultT]
        """

        def evaluator(threshold: float) -> ResultT:
            """Evaluate the metric at a specific threshold across all runs.

            Parameters
            ----------
            threshold
                Detection threshold.

            Returns
            -------
            ResultT
            """
            transformed_runs = [self.policy.apply(run, threshold) for run in runs]
            return self.source.evaluate(transformed_runs)

        return evaluator


class NoResetDerivedMetric[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any], ResultInT, ResultOutT]:
    """Wrap a classical derived metric with no-reset-aware source metrics.

    Parameters
    ----------
    source
        Formula that combines source metric values.
    bases
        Named no-reset metrics providing the source values.

    Raises
    ------
    ValueError
        If any source metric required by the derived formula is missing.
    """

    def __init__(
        self,
        source: DerivedMetric[NoResetDetectionTrace[StateT], ProviderT, ResultInT, ResultOutT],
        bases: Mapping[str, NoResetThresholdMetric[StateT, ProviderT, ResultInT]],
    ) -> None:
        missing = set(source.bases).difference(bases)
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(f"Missing no-reset bases for derived metric: {missing_names}")

        self.source = source
        self.bases = bases

    def evaluate(
        self,
        runs: Sequence[SingleRun[OnlineDetectionTrace[StateT], ProviderT]],
    ) -> Callable[[float], ResultOutT]:
        """Return a callable that evaluates the wrapped derived metric at a threshold.

        Parameters
        ----------
        runs
            Sequence of runs to evaluate.

        Returns
        -------
        Callable[[float], ResultOutT]
        """
        base_callables = {name: metric.evaluate(runs) for name, metric in self.bases.items()}

        def evaluator(threshold: float) -> ResultOutT:
            """Evaluate the derived metric at a specific threshold.

            Parameters
            ----------
            threshold
                Detection threshold.

            Returns
            -------
            ResultOutT
            """
            values = {name: metric_at_threshold(threshold) for name, metric_at_threshold in base_callables.items()}
            return self.source.compute(values)

        return evaluator


def wrap_noreset_single_run_metric[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any], ResultT](
    source: ISingleRunMetric[NoResetDetectionTrace[StateT], ProviderT, ResultT],
    policy: NoResetPolicy[StateT, ProviderT],
) -> NoResetSingleRunMetric[StateT, ProviderT, ResultT]:
    """Construct a no-reset single-run metric wrapper.

    Parameters
    ----------
    source
        Source metric operating on ``NoResetDetectionTrace``.
    policy
        No-reset policy.

    Returns
    -------
    NoResetSingleRunMetric
    """
    return NoResetSingleRunMetric(source=source, policy=policy)


def wrap_noreset_multiple_run_metric[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any], ResultT](
    source: IMultipleRunMetric[NoResetDetectionTrace[StateT], ProviderT, ResultT],
    policy: NoResetPolicy[StateT, ProviderT],
) -> NoResetMultipleRunMetric[StateT, ProviderT, ResultT]:
    """Construct a no-reset multiple-run metric wrapper.

    Parameters
    ----------
    source
        Source metric operating on ``NoResetDetectionTrace``.
    policy
        No-reset policy.

    Returns
    -------
    NoResetMultipleRunMetric
    """
    return NoResetMultipleRunMetric(source=source, policy=policy)


def wrap_noreset_derived_metric[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any], ResultInT, ResultOutT](
    base: DerivedMetric[NoResetDetectionTrace[StateT], ProviderT, ResultInT, ResultOutT],
    bases: Mapping[str, NoResetThresholdMetric[StateT, ProviderT, ResultInT]],
) -> NoResetDerivedMetric[StateT, ProviderT, ResultInT, ResultOutT]:
    """Construct a no-reset derived metric wrapper.

    Parameters
    ----------
    base
        Derived metric formula.
    bases
        Named no-reset metrics.

    Returns
    -------
    NoResetDerivedMetric
    """
    return NoResetDerivedMetric(source=base, bases=bases)
