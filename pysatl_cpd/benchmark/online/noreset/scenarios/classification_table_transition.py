# -*- coding: ascii -*-
"""Transition-filtered classification table scenario for no-reset benchmarks."""

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
from pysatl_cpd.benchmark.online.noreset.scenarios.base import DataT, NoResetBenchmarkScenario
from pysatl_cpd.benchmark.online.noreset.thresholds.resolver import NoResetThresholdResolver
from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.classification_table import (
    NoResetClassificationTableAnalyzer,
)
from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.state_arl import NoResetArlAnalyzer
from pysatl_cpd.benchmark.online.noreset.tooling.bisegment_cut import NOOP_BISEGMENT_CUT
from pysatl_cpd.benchmark.online.noreset.tooling.pickers import OnlineNoResetBisegmentByTransitionPicker
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.benchmark.scenarios import BenchmarkJob
from pysatl_cpd.core import OnlineDetectionTrace
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.data import Dataset, StateDataset, TimeseriesAnnotation
from pysatl_cpd.data.typedefs import (
    BisegmentFilter,
    BisegmentInfo,
    SegmentFilter,
    StateDescriptor,
    TransitionDescriptor,
)


class NoResetClassificationTableByTransitionScenario(NoResetBenchmarkScenario[DataT, pd.DataFrame]):
    """Scenario that computes classification metrics for a specific transition.

    Parameters
    ----------
    entries
        Detector entries to benchmark.
    collect_states
        Whether to retain algorithm states during detection (default False).
    transition
        Target transition for bisegment filtering.
    use_arl
        Whether to include an ARL column (default False).
    arl_length
        Expected length of each no-change run; required if use_arl is True.

    Raises
    ------
    ValueError
        If ``transition`` is None, or ``use_arl`` is True without a
        positive ``arl_length``.
    """

    def __init__(
        self,
        entries: Sequence[OnlineNoResetBenchmarkEntry],
        collect_states: bool = False,
        transition: TransitionDescriptor | None = None,
        use_arl: bool = False,
        arl_length: int | None = None,
    ) -> None:
        super().__init__(entries, collect_states=collect_states)
        self.transition = transition
        self.use_arl = use_arl
        self.arl_length = arl_length
        self._has_arl_providers = False
        if self.transition is None:
            raise ValueError("transition is required")
        if self.use_arl and (self.arl_length is None or self.arl_length <= 0):
            raise ValueError("use_arl=True requires a positive arl_length")
        self._threshold_resolver = NoResetThresholdResolver()
        self._classification_analyzer = NoResetClassificationTableAnalyzer()
        self._arl_analyzer = NoResetArlAnalyzer()

    def set_registry(self, registry: BenchmarkRegistry[DataT, OnlineDetectionTrace[Any]]) -> None:
        """Set the registry on both classification and ARL analyzers."""
        self._classification_analyzer.registry = registry
        self._arl_analyzer.registry = registry

    def set_classification_report(self, classification_report: NoResetClassificationReport[Any, Any]) -> None:
        """Set the classification report on the internal analyzer."""
        self._classification_analyzer.classification_report = classification_report

    def prepare_benchmark_jobs(
        self,
        dataset: Dataset[DataT, TimeseriesAnnotation],
    ) -> Sequence[BenchmarkJob[DataT]]:
        """Build jobs for bisegment providers and optionally ARL providers.

        Filters the dataset to bisegments matching the transition, and
        optionally adds no-change providers for ARL evaluation.

        Parameters
        ----------
        dataset
            Input dataset with segment and bisegment annotations.

        Returns
        -------
        Sequence[BenchmarkJob]
            Jobs per entry, each with bisegment (and optionally ARL) providers.
        """
        transition = self.transition_checked
        bisegments_filter = self._get_bisegments_filter_by_transition(transition)
        bisegments = list(dataset.filter_by_bisegments(bisegments_filter).timeseries)

        jobs: list[BenchmarkJob[DataT]] = []
        for entry in self.entries:
            detector = self._make_detector(entry)
            jobs.append(BenchmarkJob(detector, bisegments))

            if self.use_arl:
                arl_dataset = self._get_arl_dataset(dataset, transition.curr_state, self.arl_length_checked)
                self._has_arl_providers = bool(arl_dataset.timeseries)
                if arl_dataset.timeseries:
                    # ARL no-change runs must bypass detector-level crop.
                    arl_detector = self._make_detector(self._entry_for_arl(entry))
                    jobs.append(BenchmarkJob(arl_detector, arl_dataset.timeseries))

        return jobs

    def analyze(
        self,
        registry: BenchmarkRegistry[DataT, OnlineDetectionTrace[Any]],
    ) -> dict[ChangePointDetectorDescription, pd.DataFrame]:
        """Evaluate classification metrics per entry, optionally adding ARL.

        Picks runs using a transition-based picker, resolves thresholds,
        computes the classification table, and appends an ARL column if
        configured.

        Parameters
        ----------
        registry
            Registry containing cached detection runs.

        Returns
        -------
        dict[ChangePointDetectorDescription, pd.DataFrame]
            Classification table (with optional ARL column) per detector
            description.
        """
        self.set_registry(registry)
        transition = self.transition_checked
        transition_picker = OnlineNoResetBisegmentByTransitionPicker(transition)
        report = self._classification_analyzer.classification_report
        result: dict[ChangePointDetectorDescription, pd.DataFrame] = {}
        for entry in self.entries:
            runs = self._classification_analyzer.pick_runs(
                entry,
                entries_picker=transition_picker,
            )
            thresholds = self._threshold_resolver.resolve_classification_thresholds(entry, runs, report)
            classification_table = self._classification_analyzer.analyze(
                entry,
                thresholds,
                entries_picker=transition_picker,
            )
            if self.use_arl and not classification_table.empty and self._has_arl_providers:
                classification_table = classification_table.copy()
                arl_entry = self._entry_for_arl(entry)
                arl_table = self._arl_analyzer.analyze(
                    arl_entry,
                    transition.curr_state,
                    classification_table["threshold"].tolist(),
                    self.arl_length_checked,
                )
                classification_table["arl"] = arl_table["arl"]
            result[entry.description] = classification_table
        return result

    @property
    def transition_checked(self) -> TransitionDescriptor:
        """Validated transition descriptor; raises if not set."""
        if self.transition is None:
            raise ValueError("transition is required")
        return self.transition

    @property
    def arl_length_checked(self) -> int:
        """Validated ARL length; raises if not set or not positive."""
        if self.arl_length is None or self.arl_length <= 0:
            raise ValueError("use_arl=True requires a positive arl_length")
        return self.arl_length

    @staticmethod
    def _get_bisegments_filter_by_transition(transition: TransitionDescriptor) -> BisegmentFilter:
        """Build a filter that matches bisegments by their transition."""

        def filter_fn(bisegment: BisegmentInfo) -> bool:
            return (
                bisegment.transition.curr_state == transition.curr_state
                and bisegment.transition.next_state == transition.next_state
            )

        return filter_fn

    @staticmethod
    def _get_segments_filter_by_state(state: StateDescriptor) -> SegmentFilter:
        """Build a filter that matches segments by their state."""
        return lambda segment: segment.state == state

    def _get_arl_dataset(
        self,
        dataset: Dataset[DataT, TimeseriesAnnotation],
        state: StateDescriptor,
        arl_length: int,
    ) -> StateDataset[DataT]:
        """Build an ARL dataset for a given state and length.

        Creates a ``StateDataset`` of the requested length and raises if
        no no-change providers can be formed.

        Parameters
        ----------
        dataset
            Input dataset with segment annotations.
        state
            State descriptor to filter segments by.
        arl_length
            Expected length of each no-change run.

        Returns
        -------
        StateDataset
            Dataset with no-change providers of the requested length.
        """
        try:
            return StateDataset.from_dataset(dataset, arl_length, state=state)
        except ValueError:
            # Interactive callers expect ARL to be skipped cleanly when the
            # dataset cannot supply enough no-change data for the requested run length.
            return StateDataset([], state=state)

    @staticmethod
    def _entry_for_arl(entry: OnlineNoResetBenchmarkEntry) -> OnlineNoResetBenchmarkEntry:
        """Build an entry variant that disables detector-level crop for ARL runs."""
        return replace(entry, bisegment_cut=NOOP_BISEGMENT_CUT)
