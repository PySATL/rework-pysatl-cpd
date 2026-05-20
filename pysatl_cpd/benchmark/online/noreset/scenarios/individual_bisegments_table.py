# -*- coding: ascii -*-
"""Per-bisegment classification table scenario for no-reset benchmarks."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

import pandas as pd

from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry
from pysatl_cpd.benchmark.online.noreset.scenarios.base import DataT, NoResetBenchmarkScenario
from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.individual_bisegment import NoResetBisegmentAnalyzer
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.benchmark.scenarios import BenchmarkJob
from pysatl_cpd.core import OnlineDetectionTrace
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.data import Dataset, TimeseriesAnnotation


class NoResetBisegmentsTableScenario(NoResetBenchmarkScenario[DataT, pd.DataFrame]):
    """Scenario that computes per-bisegment classification at a fixed threshold.

    Parameters
    ----------
    entries
        Detector entries to benchmark.
    collect_states
        Whether to retain algorithm states during detection (default False).
    threshold
        Fixed detection threshold (default 0.0).
    """

    def __init__(
        self,
        entries: Sequence[OnlineNoResetBenchmarkEntry],
        collect_states: bool = False,
        threshold: float = 0.0,
    ) -> None:
        super().__init__(entries, collect_states=collect_states)
        self.threshold = threshold
        self._bisegment_analyzer = NoResetBisegmentAnalyzer()

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
        """Evaluate per-bisegment metrics for every entry.

        Parameters
        ----------
        registry
            Registry containing cached detection runs.

        Returns
        -------
        dict[pysatl_cpd.core.change_point_detector.ChangePointDetectorDescription, pd.DataFrame]
            Bisegment classification table per detector description.
        """
        self.set_registry(registry)
        return {entry.description: self._bisegment_analyzer.analyze(entry, self.threshold) for entry in self.entries}

    def set_registry(self, registry: BenchmarkRegistry[DataT, OnlineDetectionTrace[Any]]) -> None:
        """Set the registry used by the internal bisegment analyzer."""
        self._bisegment_analyzer.registry = registry

    def set_classification_report(self, classification_report: Any) -> None:
        """Set the classification report used by the internal bisegment analyzer."""
        self._bisegment_analyzer.classification_report = classification_report

    def handle_benchmark_error(self, job: BenchmarkJob[DataT], exc: ValueError) -> None:
        """Re-raise with a message suggesting a data-transformer fix."""
        msg = (
            "Failed to benchmark bisegments for algorithm "
            f"{job.detector.description}. "
            "Likely data shape mismatch between algorithm and providers; "
            "set entry.data_transformer to convert input format."
        )
        raise ValueError(msg) from exc
