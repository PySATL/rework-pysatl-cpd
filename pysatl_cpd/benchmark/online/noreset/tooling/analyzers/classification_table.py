# -*- coding: ascii -*-
"""Threshold-sweep classification table analyzer for no-reset benchmarks."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry

from pysatl_cpd.benchmark.online.noreset.metrics import (
    NoResetClassificationReport,
    NoResetMeanDelayMetric,
    NoResetMedianDelayMetric,
)
from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.base import (
    NoResetAnalyzerBase,
    NoResetBisigementClassificationMixin,
)
from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.individual_bisegment import NoResetBisegmentAnalyzer
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.benchmark.tooling import BenchmarkEntriesPicker
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace


class NoResetClassificationTableAnalyzer(NoResetAnalyzerBase, NoResetBisigementClassificationMixin):
    """Analyzer that computes classification metrics across a range of thresholds."""

    def _set_registry(self, registry: BenchmarkRegistry[Any, OnlineDetectionTrace[Any]]) -> None:
        """Set the registry used by this analyzer.

        Parameters
        ----------
        registry
            Registry containing cached detection runs.
        """
        self._registry = registry

    def _set_classification_report(
        self,
        classification_report: NoResetClassificationReport[Any, Any],
    ) -> None:
        """Set the classification report used by this analyzer.

        Parameters
        ----------
        classification_report
            Report defining TP/FP/FN source metrics and derived metrics.
        """
        self._classification_report = classification_report

    def analyze(
        self,
        benchmark_entry: OnlineNoResetBenchmarkEntry,
        thresholds: Sequence[float],
        *,
        entries_picker: BenchmarkEntriesPicker | None = None,
    ) -> pd.DataFrame:
        """Evaluate classification metrics at every threshold in a sweep.

        For each threshold, computes TP, FP, FN, precision, recall, F1,
        mean delay, and median delay across all runs picked for the entry.

        Parameters
        ----------
        benchmark_entry
            Entry describing detector configuration used for picking and validation.
        thresholds
            Threshold values to evaluate.
        entries_picker
            Override picker for selecting registry entries.

        Returns
        -------
        pd.DataFrame
            Table with one row per threshold containing all metrics.
        """
        report = self.classification_report
        runs = self.pick_runs(
            benchmark_entry,
            entries_picker=entries_picker,
        )

        base_columns = [
            "threshold",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "mean_delay",
            "median_delay",
        ]
        if not runs:
            return pd.DataFrame(columns=base_columns)

        self.validate_bisegment_runs(runs, benchmark_entry.description)
        report_fn = report.evaluate(runs)

        rows: list[dict[str, float | int]] = []
        for threshold in tqdm(thresholds, desc="Evaluating thresholds", unit="threshold"):
            metrics = report_fn(float(threshold))
            row: dict[str, float | int] = {"threshold": float(threshold)}
            row.update(metrics)
            rows.append(row)

        frame = pd.DataFrame(rows)
        max_delay = NoResetBisegmentAnalyzer.extract_error_margin_from_report(report)[1]
        mean_delay_eval = NoResetMeanDelayMetric(max_delay=max_delay, strict=True).evaluate(cast(Any, runs))
        median_delay_eval = NoResetMedianDelayMetric(max_delay=max_delay, strict=True).evaluate(cast(Any, runs))
        frame["mean_delay"] = [float(mean_delay_eval(float(th))) for th in frame["threshold"]]
        frame["median_delay"] = [float(median_delay_eval(float(th))) for th in frame["threshold"]]

        return frame
