# -*- coding: ascii -*-
"""ARL-by-state scenario for no-reset benchmarks."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pandas as pd

from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry
from pysatl_cpd.benchmark.online.noreset.metrics import NoResetClassificationReport
from pysatl_cpd.benchmark.online.noreset.scenarios.base import DataT, NoResetBenchmarkScenario, logger
from pysatl_cpd.benchmark.online.noreset.thresholds.resolver import NoResetThresholdResolver
from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.state_arl import NoResetArlAnalyzer
from pysatl_cpd.benchmark.online.noreset.tooling.bisegment_cut import NOOP_BISEGMENT_CUT
from pysatl_cpd.benchmark.online.noreset.tooling.pickers import OnlineNoResetNoChangeByStatePicker
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.benchmark.scenarios import BenchmarkJob
from pysatl_cpd.core import OnlineDetectionTrace
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.data import Dataset, StateDataset, TimeseriesAnnotation
from pysatl_cpd.data.typedefs import SegmentFilter, StateDescriptor


class NoResetArlByStateScenario(NoResetBenchmarkScenario[DataT, pd.DataFrame]):
    """Scenario that evaluates ARL for a specific state across thresholds.

    Parameters
    ----------
    entries
        Detector entries to benchmark.
    collect_states
        Whether to retain algorithm states during detection (default False).
    state
        Target state for no-change providers.
    arl_length
        Expected length of each no-change run. Must be positive.

    Raises
    ------
    ValueError
        If ``state`` is None or ``arl_length`` is not positive.
    """

    def __init__(
        self,
        entries: Sequence[OnlineNoResetBenchmarkEntry],
        collect_states: bool = False,
        state: StateDescriptor | None = None,
        arl_length: int = 0,
    ) -> None:
        super().__init__(entries, collect_states=collect_states)
        self.state = state
        self.arl_length = arl_length
        self._has_arl_providers = False
        self._threshold_resolver = NoResetThresholdResolver()
        self._arl_analyzer = NoResetArlAnalyzer()
        if self.state is None:
            raise ValueError("state is required")
        if self.arl_length <= 0:
            raise ValueError(f"arl_length must be positive, got {self.arl_length}")

    def prepare_benchmark_jobs(
        self,
        dataset: Dataset[DataT, TimeseriesAnnotation],
    ) -> Sequence[BenchmarkJob[DataT]]:
        """Build ARL benchmark jobs for each entry.

        Filters the dataset to segments matching the target state,
        creates a no-change provider dataset of the requested length,
        and builds one job per entry.

        Parameters
        ----------
        dataset
            Input dataset with segment annotations.

        Returns
        -------
        Sequence[BenchmarkJob]
            One job per entry, each using the no-change providers.

        Raises
        ------
        ValueError
            If no segments match the target state.
        """
        segments_filter = self._get_segments_filter_by_state(self.state_checked)
        segments_dataset = dataset.filter_by_segments(segments_filter)
        if not segments_dataset.timeseries:
            raise ValueError(f"No segments in state {self.state_checked}")

        arl_dataset = StateDataset.from_dataset(dataset, self.arl_length, state=self.state_checked)
        self._has_arl_providers = bool(arl_dataset.timeseries)
        if not arl_dataset.timeseries:
            logger.warning(
                "Merged no-change provider length %s is less than arl_length=%s; no ARL runs will be registered.",
                len(segments_dataset.merge()),
                self.arl_length,
            )

        # ARL must run on untrimmed no-change providers even when benchmark entries
        # carry a non-noop bisegment_cut for bisegment scenarios.
        return [
            BenchmarkJob(self._make_detector(self._entry_for_arl(entry)), arl_dataset.timeseries)
            for entry in self.entries
        ]

    def set_registry(self, registry: BenchmarkRegistry[DataT, OnlineDetectionTrace[Any]]) -> None:
        """Set the registry used by the internal ARL analyzer."""
        self._arl_analyzer.registry = registry

    def set_classification_report(self, classification_report: NoResetClassificationReport[Any, Any]) -> None:
        """No-op; ARL scenario does not require a classification report."""
        del classification_report

    def analyze(
        self,
        registry: BenchmarkRegistry[DataT, OnlineDetectionTrace[Any]],
    ) -> dict[ChangePointDetectorDescription, pd.DataFrame]:
        """Evaluate ARL for every entry using resolved thresholds.

        Parameters
        ----------
        registry
            Registry containing cached detection runs.

        Returns
        -------
        dict[ChangePointDetectorDescription, pd.DataFrame]
            ARL table per detector description, or empty dict if no
            ARL providers were registered.
        """
        if not self._has_arl_providers:
            return {}
        self.set_registry(registry)
        result: dict[ChangePointDetectorDescription, pd.DataFrame] = {}
        state = self.state_checked
        arl_length = self.arl_length
        picker = OnlineNoResetNoChangeByStatePicker(state)
        for entry in self.entries:
            arl_entry = self._entry_for_arl(entry)
            runs = [
                run
                for run in self._arl_analyzer.pick_runs(
                    arl_entry,
                    entries_picker=picker,
                )
                if len(run.provider) == arl_length
            ]
            self._arl_analyzer.validate_arl_runs(runs, arl_entry.description, state, arl_length)
            thresholds = self._threshold_resolver.resolve_arl_thresholds(arl_entry, runs, None)
            result[entry.description] = self._arl_analyzer.analyze_runs(runs, thresholds)
        return result

    @property
    def state_checked(self) -> StateDescriptor:
        """Validated state descriptor; raises if not set."""
        if self.state is None:
            raise ValueError("state is required")
        return self.state

    @staticmethod
    def _get_segments_filter_by_state(state: StateDescriptor) -> SegmentFilter:
        """Build a filter that matches segments by their state."""
        return lambda segment: segment.state == state

    @staticmethod
    def _entry_for_arl(entry: OnlineNoResetBenchmarkEntry) -> OnlineNoResetBenchmarkEntry:
        """Build an entry variant that disables detector-level crop for ARL runs."""
        return replace(entry, bisegment_cut=NOOP_BISEGMENT_CUT)
