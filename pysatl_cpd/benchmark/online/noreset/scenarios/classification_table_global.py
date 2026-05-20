# -*- coding: ascii -*-
"""Global classification table scenario for no-reset benchmarks."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

import pandas as pd

from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry
from pysatl_cpd.benchmark.online.noreset.scenarios.base import DataT, NoResetBenchmarkScenario
from pysatl_cpd.benchmark.online.noreset.thresholds.resolver import NoResetThresholdResolver
from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.classification_table import (
    NoResetClassificationTableAnalyzer,
)
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.benchmark.scenarios import BenchmarkJob
from pysatl_cpd.core import OnlineDetectionTrace
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.data import Dataset, TimeseriesAnnotation


class NoResetClassificationTableScenario(NoResetBenchmarkScenario[DataT, pd.DataFrame]):
    """Scenario that computes classification metrics across all transitions.

    Parameters
    ----------
    entries
        Detector entries to benchmark.
    collect_states
        Whether to retain algorithm states during detection (default False).
    """

    def __init__(
        self,
        entries: Sequence[OnlineNoResetBenchmarkEntry],
        collect_states: bool = False,
    ) -> None:
        super().__init__(entries, collect_states=collect_states)
        self._threshold_resolver = NoResetThresholdResolver()
        self._classification_analyzer = NoResetClassificationTableAnalyzer()

    def set_registry(self, registry: BenchmarkRegistry[DataT, OnlineDetectionTrace[Any]]) -> None:
        """Set the registry used by the internal classification analyzer."""
        self._classification_analyzer.registry = registry

    def set_classification_report(self, classification_report: Any) -> None:
        """Set the classification report used by the internal analyzer."""
        self._classification_analyzer.classification_report = classification_report

    def prepare_benchmark_jobs(
        self,
        dataset: Dataset[DataT, TimeseriesAnnotation],
    ) -> Sequence[BenchmarkJob[DataT]]:
        """Build benchmark jobs using only bisegment providers.

        Parameters
        ----------
        dataset
            Input dataset with bisegment annotations.

        Returns
        -------
        Sequence[BenchmarkJob]
            One job per entry, each using the bisegment providers.
        """
        bisegments = list(dataset.filter_by_bisegments().timeseries)
        return [BenchmarkJob(self._make_detector(entry), bisegments) for entry in self.entries]

    def analyze(
        self,
        registry: BenchmarkRegistry[DataT, OnlineDetectionTrace[Any]],
    ) -> dict[ChangePointDetectorDescription, pd.DataFrame]:
        """Evaluate classification metrics across resolved thresholds.

        Resolves thresholds from picked runs, then evaluates the
        classification table for each entry.

        Parameters
        ----------
        registry
            Registry containing cached detection runs.

        Returns
        -------
        dict[ChangePointDetectorDescription, pd.DataFrame]
            Classification table per detector description.
        """
        self.set_registry(registry)
        report = self._classification_analyzer.classification_report
        result: dict[ChangePointDetectorDescription, pd.DataFrame] = {}
        for entry in self.entries:
            runs = self._classification_analyzer.pick_runs(
                entry,
            )
            thresholds = self._threshold_resolver.resolve_classification_thresholds(entry, runs, report)
            result[entry.description] = self._classification_analyzer.analyze(
                entry,
                thresholds,
            )
        return result
