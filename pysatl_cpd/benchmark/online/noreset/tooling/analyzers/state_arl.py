# -*- coding: ascii -*-
"""ARL (average run length) analyzer for no-reset benchmarks."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from pysatl_cpd.benchmark.online.noreset.metrics import NoResetARLMetric

if TYPE_CHECKING:
    from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry
from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.base import NoResetAnalyzerBase
from pysatl_cpd.benchmark.online.noreset.tooling.pickers import OnlineNoResetNoChangeByStatePicker
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data import LabeledData, TimeseriesAnnotation
from pysatl_cpd.data.typedefs import ProviderType, StateDescriptor


class NoResetArlAnalyzer(NoResetAnalyzerBase):
    """Analyzer that computes average run length at a range of thresholds."""

    def _set_registry(self, registry: BenchmarkRegistry[Any, OnlineDetectionTrace[Any]]) -> None:
        """Set the registry used by this analyzer.

        Parameters
        ----------
        registry
            Registry containing cached detection runs.
        """
        self._registry = registry

    def analyze(
        self,
        benchmark_entry: OnlineNoResetBenchmarkEntry,
        state: StateDescriptor,
        thresholds: Sequence[float],
        arl_length: int,
    ) -> pd.DataFrame:
        """Compute ARL for each threshold value.

        Retrieves no-change runs matching the detector, state, and
        length, then evaluates the ARL metric at each threshold.

        Parameters
        ----------
        benchmark_entry
            Entry describing detector configuration used for picking and validation.
        state
            State descriptor for the no-change runs.
        thresholds
            Threshold values to evaluate.
        arl_length
            Expected length of each no-change run. Must be positive.

        Returns
        -------
        pd.DataFrame
            Table with columns ``threshold`` and ``arl``.

        Raises
        ------
        ValueError
            If ``arl_length`` is not positive.
        """
        if arl_length <= 0:
            raise ValueError(f"arl_length must be positive, got {arl_length}")

        picker = OnlineNoResetNoChangeByStatePicker(state)
        runs = [
            run
            for run in self.pick_runs(
                benchmark_entry,
                entries_picker=picker,
            )
            if len(run.provider) == arl_length
        ]
        self.validate_arl_runs(runs, benchmark_entry.description, state, arl_length)

        return self.analyze_runs(runs, thresholds)

    def analyze_runs(
        self,
        runs: Sequence[SingleRun[OnlineDetectionTrace[Any], LabeledData[Any, TimeseriesAnnotation]]],
        thresholds: Sequence[float],
    ) -> pd.DataFrame:
        """Compute ARL for already selected and validated no-change runs."""
        arl_eval = NoResetARLMetric(strict=True).evaluate(cast(Any, runs))
        rows = [{"threshold": float(t), "arl": float(arl_eval(t))} for t in thresholds]
        return pd.DataFrame(rows, columns=["threshold", "arl"])

    @staticmethod
    def validate_arl_runs(
        runs: Sequence[SingleRun[OnlineDetectionTrace[Any], LabeledData[Any, TimeseriesAnnotation]]],
        expected_detector: ChangePointDetectorDescription,
        state: StateDescriptor,
        arl_length: int,
    ) -> None:
        """Verify that every run matches the expected detector, state, and length."""
        if not runs:
            raise ValueError("No no-change (ARL) runs found in registry for this algorithm and state")
        for run in runs:
            if run.trace.detector_description != expected_detector:
                raise ValueError("Picked ARL run does not match benchmark entry detector")
            if getattr(run.provider.annotation, "provider_type", None) != ProviderType.NO_CHANGE:
                raise ValueError("ARL analysis requires no-change providers")
            if getattr(run.provider.annotation, "state", None) != state:
                raise ValueError("Picked ARL run does not match requested state")
            if len(run.provider) != arl_length:
                raise ValueError("Picked ARL run does not match requested length")
