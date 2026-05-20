# -*- coding: ascii -*-
"""No-reset benchmark orchestrator."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from pysatl_cpd.benchmark.benchmark import Benchmark
from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry
from pysatl_cpd.benchmark.online.noreset.metrics import NoResetClassificationReport
from pysatl_cpd.benchmark.online.noreset.metrics.policy.bisegment import (
    BisegmentPolicyBase,
    EventBasedPolicy,
    MixedPolicy,
    PointBasedPolicy,
)
from pysatl_cpd.benchmark.online.noreset.policy_kind import NoResetPolicyKind
from pysatl_cpd.benchmark.online.noreset.scenarios import (
    NoResetArlByStateScenario,
    NoResetBisegmentsTableScenario,
    NoResetClassificationTableByTransitionScenario,
    NoResetClassificationTableScenario,
)
from pysatl_cpd.benchmark.registry import DEFAULT_JOB_PARALLEL_BACKEND, BenchmarkRegistry
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.online import OnlineDetectionTrace
from pysatl_cpd.core.online.ionline_algorithm import (
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
)
from pysatl_cpd.data import TimeseriesAnnotation
from pysatl_cpd.data.dataset import Dataset
from pysatl_cpd.data.providers.labeled import LabeledData
from pysatl_cpd.data.typedefs import (
    StateDescriptor,
    TransitionDescriptor,
)

type DetectorDescription = ChangePointDetectorDescription
type ClassificationTable = pd.DataFrame
type BisegmentsClassificationTable = pd.DataFrame
type ARLTable = pd.DataFrame
type SegmentState = StateDescriptor


class OnlineNoResetBenchmark[
    DataT,
    ConfigurationT: OnlineAlgorithmConfiguration,
    StateT: OnlineAlgorithmState,
](Benchmark[DataT, OnlineDetectionTrace[Any]]):
    """Benchmark subclass specialised for no-reset online detectors.

    Provides convenience methods for all no-reset scenario types:
    classification table (global and by transition), bisegments table,
    ARL by state, and PR-AUC computation.

    Classification semantics are fixed at construction via ``max_delay`` and
    policy kind selectors; use :meth:`build_classification_report` to build the
    same report configuration without instantiating a benchmark.

    Parameters
    ----------
    dataset
        Labeled dataset whose providers serve as detector inputs.
    registry
        Registry that caches per-detector execution results.
    n_jobs
        Number of parallel worker processes (default 1). Must be non-zero.
    max_delay
        Maximum delay (steps) after the true change point used by bisegment
        policies and default right tolerance in ``error_margin`` when
        ``error_margin`` is omitted.
    global_policy
        Default bisegment policy kind for TP/FP/FN and as the fallback for
        precision/recall unless overridden.
    precision_policy
        Optional policy kind override for the precision metric (TP/FP bases).
    recall_policy
        Optional policy kind override for the recall metric (TP/FN bases).
    error_margin
        ``(left, right)`` tolerance for underlying classification metrics. When
        omitted, ``(0, max_delay)`` is used.
    policy_strict
        Whether policies compare detection values to the threshold with strict
        inequality (default True).
    """

    def __init__(
        self,
        dataset: Dataset[DataT, TimeseriesAnnotation],
        registry: BenchmarkRegistry[DataT, OnlineDetectionTrace[Any]],
        *,
        n_jobs: int = 1,
        max_delay: int,
        global_policy: NoResetPolicyKind,
        precision_policy: NoResetPolicyKind | None = None,
        recall_policy: NoResetPolicyKind | None = None,
        error_margin: tuple[int, int] | None = None,
        policy_strict: bool = True,
    ) -> None:
        super().__init__(dataset, registry, n_jobs=n_jobs)
        self._classification_report: NoResetClassificationReport[StateT, LabeledData[DataT, TimeseriesAnnotation]] = (
            OnlineNoResetBenchmark.build_classification_report(
                max_delay=max_delay,
                global_policy=global_policy,
                precision_policy=precision_policy,
                recall_policy=recall_policy,
                error_margin=error_margin,
                policy_strict=policy_strict,
            )
        )

    @staticmethod
    def _policy_from_kind(
        kind: NoResetPolicyKind,
        *,
        max_delay: int,
        strict: bool,
    ) -> BisegmentPolicyBase[Any, Any]:
        if kind is NoResetPolicyKind.POINT:
            return PointBasedPolicy(max_delay=max_delay, strict=strict)
        if kind is NoResetPolicyKind.EVENT:
            return EventBasedPolicy(max_delay=max_delay, strict=strict)
        if kind is NoResetPolicyKind.MIXED:
            return MixedPolicy(max_delay=max_delay, strict=strict)
        raise NotImplementedError(f"Unknown policy kind: {kind}")

    @staticmethod
    def build_classification_report(
        *,
        max_delay: int,
        global_policy: NoResetPolicyKind,
        precision_policy: NoResetPolicyKind | None = None,
        recall_policy: NoResetPolicyKind | None = None,
        error_margin: tuple[int, int] | None = None,
        policy_strict: bool = True,
    ) -> NoResetClassificationReport[Any, Any]:
        """Build a :class:`NoResetClassificationReport` from policy kinds and delay."""
        resolved_error_margin = error_margin if error_margin is not None else (0, max_delay)
        global_bisegment_policy = OnlineNoResetBenchmark._policy_from_kind(
            global_policy, max_delay=max_delay, strict=policy_strict
        )
        precision_bisegment_policy = (
            OnlineNoResetBenchmark._policy_from_kind(precision_policy, max_delay=max_delay, strict=policy_strict)
            if precision_policy is not None
            else None
        )
        recall_bisegment_policy = (
            OnlineNoResetBenchmark._policy_from_kind(recall_policy, max_delay=max_delay, strict=policy_strict)
            if recall_policy is not None
            else None
        )
        return NoResetClassificationReport(
            error_margin=resolved_error_margin,
            global_policy=global_bisegment_policy,
            precision_policy=precision_bisegment_policy,
            recall_policy=recall_bisegment_policy,
        )

    def get_classification_table(
        self,
        entries: Sequence[OnlineNoResetBenchmarkEntry],
        *,
        collect_states: bool = False,
        n_jobs: int | None = None,
        backend: str = DEFAULT_JOB_PARALLEL_BACKEND,
    ) -> Mapping[DetectorDescription, ClassificationTable]:
        """Run a global classification-table scenario across all entries.

        Parameters
        ----------
        entries
            Detector entries to benchmark.
        collect_states
            Whether to retain algorithm states during detection (default False).
        n_jobs
            Worker count override; falls back to instance n_jobs when None.
        backend
            Joblib parallel backend identifier (default ``"loky"``).

        Returns
        -------
        Mapping[DetectorDescription, ClassificationTable]
            Classification table per detector description.
        """
        scenario: NoResetClassificationTableScenario = NoResetClassificationTableScenario(
            entries,
            collect_states=collect_states,
        )
        scenario.set_registry(self._registry)
        scenario.set_classification_report(self._classification_report)
        return self.run_scenario(scenario, n_jobs=n_jobs, backend=backend)

    def get_classification_table_by_transition(
        self,
        entries: Sequence[OnlineNoResetBenchmarkEntry],
        transition: TransitionDescriptor,
        use_arl: bool,
        arl_length: int | None = None,
        *,
        collect_states: bool = False,
        n_jobs: int | None = None,
        backend: str = DEFAULT_JOB_PARALLEL_BACKEND,
    ) -> Mapping[DetectorDescription, ClassificationTable]:
        """Run a transition-filtered classification-table scenario.

        Parameters
        ----------
        entries
            Detector entries to benchmark.
        transition
            Target transition for bisegment filtering.
        use_arl
            Whether to include an ARL column.
        arl_length
            Expected length of each no-change run; required if use_arl is True.
        collect_states
            Whether to retain algorithm states during detection (default False).
        n_jobs
            Worker count override; falls back to instance n_jobs when None.
        backend
            Joblib parallel backend identifier (default ``"loky"``).

        Returns
        -------
        Mapping[DetectorDescription, ClassificationTable]
            Classification table (with optional ARL column) per detector
            description.
        """
        scenario: NoResetClassificationTableByTransitionScenario = NoResetClassificationTableByTransitionScenario(
            entries,
            collect_states=collect_states,
            transition=transition,
            use_arl=use_arl,
            arl_length=arl_length,
        )
        scenario.set_registry(self._registry)
        scenario.set_classification_report(self._classification_report)
        return self.run_scenario(scenario, n_jobs=n_jobs, backend=backend)

    @staticmethod
    def get_pr_auc_table(
        classification_tables: Mapping[DetectorDescription, ClassificationTable],
    ) -> Mapping[DetectorDescription, float]:
        """Compute PR-AUC from classification tables using trapezoidal integration.

        Sorts by recall ascending, drops duplicate recall rows keeping the
        highest precision, and computes the area under the precision-recall
        curve via ``numpy.trapezoid``.

        Parameters
        ----------
        classification_tables
            Mapping of detector descriptions to DataFrames containing
            ``recall`` and ``precision`` columns.

        Returns
        -------
        Mapping[DetectorDescription, float]
            PR-AUC score per detector description. Entries with empty
            tables yield NaN.

        Raises
        ------
        ValueError
            If any table is missing the required ``recall`` or ``precision``
            columns.
        """
        result: dict[DetectorDescription, float] = {}
        for detector_description, table in classification_tables.items():
            pr_auc = float("nan")
            if table.empty:
                continue

            required_columns = {"recall", "precision"}
            missing_columns = required_columns.difference(table.columns)
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"Classification table must contain columns: {missing}")

            pr_data = (
                table[["recall", "precision"]]
                .sort_values(by=["recall", "precision"], ascending=[True, False])
                .drop_duplicates(subset=["recall"], keep="first")
            )

            boundary_points = pd.DataFrame([{"recall": 0.0, "precision": 1.0}, {"recall": 1.0, "precision": 0.0}])
            pr_data = pd.concat([pr_data, boundary_points], ignore_index=True).sort_values(by="recall")
            pr_auc = float(np.trapezoid(pr_data["precision"], pr_data["recall"]))
            result[detector_description] = pr_auc

        return result

    def get_ARL_table_by_state(
        self,
        entries: Sequence[OnlineNoResetBenchmarkEntry],
        state: SegmentState,
        arl_length: int,
        *,
        collect_states: bool = False,
        n_jobs: int | None = None,
        backend: str = DEFAULT_JOB_PARALLEL_BACKEND,
    ) -> Mapping[DetectorDescription, ARLTable]:
        """Run an ARL-by-state scenario.

        Parameters
        ----------
        entries
            Detector entries to benchmark.
        state
            Target state for no-change providers.
        arl_length
            Expected length of each no-change run.
        collect_states
            Whether to retain algorithm states during detection (default False).
        n_jobs
            Worker count override; falls back to instance n_jobs when None.
        backend
            Joblib parallel backend identifier (default ``"loky"``).

        Returns
        -------
        Mapping[DetectorDescription, ARLTable]
            ARL table per detector description.
        """
        scenario: NoResetArlByStateScenario = NoResetArlByStateScenario(
            entries,
            collect_states=collect_states,
            state=state,
            arl_length=arl_length,
        )
        scenario.set_registry(self._registry)
        scenario.set_classification_report(self._classification_report)
        return self.run_scenario(scenario, n_jobs=n_jobs, backend=backend)

    def get_bisegments_table(
        self,
        entries: Sequence[OnlineNoResetBenchmarkEntry],
        threshold: float,
        *,
        collect_states: bool = False,
        n_jobs: int | None = None,
        backend: str = DEFAULT_JOB_PARALLEL_BACKEND,
    ) -> Mapping[DetectorDescription, BisegmentsClassificationTable]:
        """Run a per-bisegment classification scenario at a fixed threshold.

        Parameters
        ----------
        entries
            Detector entries to benchmark.
        threshold
            Detection threshold to apply.
        collect_states
            Whether to retain algorithm states during detection (default False).
        n_jobs
            Worker count override; falls back to instance n_jobs when None.
        backend
            Joblib parallel backend identifier (default ``"loky"``).

        Returns
        -------
        Mapping[DetectorDescription, BisegmentsClassificationTable]
            Bisegment classification table per detector description.
        """
        scenario: NoResetBisegmentsTableScenario = NoResetBisegmentsTableScenario(
            entries,
            collect_states=collect_states,
            threshold=threshold,
        )
        scenario.set_registry(self._registry)
        scenario.set_classification_report(self._classification_report)
        return self.run_scenario(scenario, n_jobs=n_jobs, backend=backend)
