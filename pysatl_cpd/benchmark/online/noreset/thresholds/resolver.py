# -*- coding: ascii -*-
"""Threshold resolution logic for no-reset benchmarks."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry
from pysatl_cpd.benchmark.online.noreset.metrics import NoResetClassificationReport
from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import AutoThresholdsRange
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.providers.labeled import LabeledData


@dataclass(frozen=True, kw_only=True)
class ThresholdAutoTuneConfig:
    """Configuration for automatic threshold selection."""

    threshold_count: int = 100
    bs_error: float = 1e-6
    t_min: float = 0.0
    t_max: float = 1.0


@dataclass(frozen=True, kw_only=True)
class ThresholdAutoTuneResult:
    """Result of automatic threshold tuning."""

    min_threshold: float
    max_threshold: float
    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray


def _binary_search_boundary(
    *,
    t_min: float,
    t_max: float,
    eps: float,
    go_right_if_true: Callable[[float], bool],
    return_right: bool,
) -> float:
    """Locate a decision boundary within a threshold interval via binary search."""
    left = float(t_min)
    right = float(t_max)

    while (right - left) > eps:
        mid = 0.5 * (left + right)
        if go_right_if_true(mid):
            left = mid
        else:
            right = mid

    return right if return_right else left


def auto_pick_thresholds[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](
    runs: Sequence[SingleRun[OnlineDetectionTrace[StateT], ProviderT]],
    make_recall: Callable[
        [Sequence[SingleRun[OnlineDetectionTrace[StateT], ProviderT]]],
        Callable[[float], float],
    ],
    make_precision: Callable[
        [Sequence[SingleRun[OnlineDetectionTrace[StateT], ProviderT]]],
        Callable[[float], float],
    ],
    *,
    config: ThresholdAutoTuneConfig,
) -> ThresholdAutoTuneResult:
    """Automatically select meaningful threshold bounds."""
    if config.threshold_count < 2:
        raise ValueError("threshold_count must be >= 2")
    if not np.isfinite(config.t_min) or not np.isfinite(config.t_max):
        raise ValueError("t_min and t_max must be finite")
    if config.t_max <= config.t_min:
        raise ValueError("t_max must be > t_min")
    if config.bs_error <= 0:
        raise ValueError("bs_error must be > 0")

    recall_fn = make_recall(runs)
    precision_fn = make_precision(runs)
    eps = config.bs_error

    final_max_threshold = _binary_search_boundary(
        t_min=config.t_min,
        t_max=config.t_max,
        eps=eps,
        go_right_if_true=lambda t: precision_fn(t) < (1.0 - eps) or recall_fn(t) > eps,
        return_right=True,
    )

    final_min_threshold = _binary_search_boundary(
        t_min=config.t_min,
        t_max=config.t_max,
        eps=eps,
        go_right_if_true=lambda t: precision_fn(t) < eps and recall_fn(t) > (1.0 - eps),
        return_right=False,
    )

    if final_min_threshold > final_max_threshold:
        final_min_threshold, final_max_threshold = final_max_threshold, final_min_threshold  # pragma: no cover

    thresholds = np.linspace(
        final_min_threshold,
        final_max_threshold,
        config.threshold_count,
        dtype=np.float64,
    )

    precision = np.array([precision_fn(float(t)) for t in thresholds], dtype=np.float64)
    recall = np.array([recall_fn(float(t)) for t in thresholds], dtype=np.float64)

    return ThresholdAutoTuneResult(
        min_threshold=float(final_min_threshold),
        max_threshold=float(final_max_threshold),
        thresholds=thresholds,
        precision=precision,
        recall=recall,
    )


@dataclass
class NoResetThresholdResolver:
    """Resolves threshold grids for classification and ARL evaluation."""

    def resolve_classification_thresholds(
        self,
        entry: OnlineNoResetBenchmarkEntry,
        runs: Sequence[SingleRun[OnlineDetectionTrace[Any], Any]],
        report: NoResetClassificationReport[Any, Any],
    ) -> list[float]:
        """Resolve classification thresholds from an entry and its runs.

        Uses the entry's explicit threshold range when available,
        otherwise auto-tunes using the report's precision and recall
        metrics.

        Parameters
        ----------
        entry
            Benchmark entry with a threshold range specification.
        runs
            Picked runs used for auto-tuning when applicable.
        report
            Classification report providing precision and recall metrics.

        Returns
        -------
        list[float]
            Sorted list of threshold values.
        """
        if not isinstance(entry.thresholds, AutoThresholdsRange):
            return [float(t) for t in entry.thresholds.thresholds_range]

        precision_metric = report.bases["precision"]
        recall_metric = report.bases["recall"]

        t_max = self.infer_t_max_from_trace_values(runs)
        config = ThresholdAutoTuneConfig(
            threshold_count=entry.thresholds.count,
            t_min=0.0,
            t_max=t_max,
        )
        auto_result = auto_pick_thresholds(
            runs=runs,
            make_recall=lambda x: cast(Any, recall_metric).evaluate(x),
            make_precision=lambda x: cast(Any, precision_metric).evaluate(x),
            config=config,
        )
        return [float(t) for t in auto_result.thresholds.tolist()]

    def resolve_arl_thresholds(
        self,
        entry: OnlineNoResetBenchmarkEntry,
        runs: Sequence[SingleRun[OnlineDetectionTrace[Any], Any]],
        thresholds: Sequence[float] | None,
    ) -> list[float]:
        """Resolve ARL thresholds from an entry and its no-change runs.

        Uses the provided thresholds when given, otherwise infers a
        range from the minimum and maximum detection function values
        across all runs.

        Parameters
        ----------
        entry
            Benchmark entry with a threshold range specification.
        runs
            No-change runs used to infer the threshold range.
        thresholds
            Optional explicit threshold list to use directly.

        Returns
        -------
        list[float]
            Sorted list of threshold values.

        Raises
        ------
        ValueError
            If no runs are provided or any run has an empty trace.
        """
        if thresholds is not None and len(thresholds) > 0:
            return [float(t) for t in thresholds]

        min_values: list[float] = []
        max_values: list[float] = []
        for run in runs:
            values = run.trace.detection_function
            if len(values) == 0:
                raise ValueError("Cannot infer ARL thresholds: empty detection trace")
            finite_values = values[np.isfinite(values)]
            if len(finite_values) == 0:
                raise ValueError("Cannot infer ARL thresholds: no finite detection values in run")
            min_values.append(float(np.min(finite_values)))
            max_values.append(float(np.max(finite_values)))

        if not min_values:
            raise ValueError("Cannot infer ARL thresholds: no runs provided")

        t_min, t_max = min(min_values), max(max_values)

        if isinstance(entry.thresholds, AutoThresholdsRange):
            threshold_count = entry.thresholds.count
        else:
            threshold_count = len(entry.thresholds.thresholds_range)
            if threshold_count < 1:
                raise ValueError("Cannot auto-pick ARL thresholds: entry has empty thresholds_range")

        if threshold_count == 1:
            return [float(0.5 * (t_min + t_max))]

        return [float(x) for x in np.linspace(t_min, t_max, threshold_count, dtype=np.float64)]

    @staticmethod
    def infer_t_max_from_trace_values(
        runs: Sequence[SingleRun[OnlineDetectionTrace[Any], Any]],
    ) -> float:
        """Infer an upper bound for auto-tuning from trace detection values.

        Computes 101 % of the maximum detection function value across
        all runs. Returns a small positive epsilon if the result is
        non-positive.

        Parameters
        ----------
        runs
            Runs whose detection functions are scanned for the maximum.

        Returns
        -------
        float
            Upper threshold bound for auto-tuning.

        Raises
        ------
        ValueError
            If no runs are provided or any run has an empty trace.
        """
        trace_max_values: list[float] = []
        for run in runs:
            values = run.trace.detection_function
            if len(values) == 0:
                raise ValueError("Cannot infer t_max: empty detection trace")
            finite_values = values[np.isfinite(values)]
            if len(finite_values) == 0:
                raise ValueError("Cannot infer t_max: no finite detection values in run")
            trace_max_values.append(float(np.max(finite_values)))

        if not trace_max_values:
            raise ValueError("Cannot infer t_max: no runs provided")

        t_max = 1.01 * max(trace_max_values)
        if t_max <= 0.0:
            return 1e-12
        return t_max
