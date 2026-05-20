# -*- coding: ascii -*-
"""Per-bisegment classification analyzer for no-reset benchmarks."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pandas as pd

from pysatl_cpd.analysis.metrics.single_run import FalseNegativeCount, FalsePositiveCount, TruePositiveCount
from pysatl_cpd.benchmark.online.noreset.metrics import NoResetClassificationReport, wrap_noreset_single_run_metric

if TYPE_CHECKING:
    from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry

from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.base import (
    NoResetAnalyzerBase,
    NoResetBisigementClassificationMixin,
)
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.benchmark.tooling import BenchmarkEntriesPicker
from pysatl_cpd.core import OnlineDetectionTrace
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data import BisegmentAnnotation


class NoResetBisegmentAnalyzer(NoResetAnalyzerBase, NoResetBisigementClassificationMixin):
    """Analyzer that computes per-bisegment classification metrics at a fixed threshold."""

    BISEGMENTS_COLUMNS: ClassVar[tuple[str, ...]] = (
        "bisegment_name",
        "source",
        "transition",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
    )

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
        threshold: float,
        *,
        entries_picker: BenchmarkEntriesPicker | None = None,
    ) -> pd.DataFrame:
        """Compute TP, FP, FN, precision, recall, and F1 per bisegment.

        Picks runs matching the entry, validates them as bisegments,
        evaluates single-run metrics using the policies stored in the
        classification report, and returns a DataFrame with one row
        per bisegment.

        Parameters
        ----------
        benchmark_entry
            Entry describing detector configuration used for picking and validation.
        threshold
            Detection threshold to apply.
        entries_picker
            Override picker for selecting registry entries.

        Returns
        -------
        pd.DataFrame
            Table with columns: bisegment_id, original_timeseries_path,
            transition, change_point, tp, fp, fn, precision, recall, f1.
        """
        report = self.classification_report
        runs = self.pick_runs(
            benchmark_entry,
            entries_picker=entries_picker,
        )
        if not runs:
            return pd.DataFrame(columns=list(self.BISEGMENTS_COLUMNS))

        self.validate_bisegment_runs(runs, benchmark_entry.description)
        error_margin = self.extract_error_margin_from_report(report)
        metric_evaluators = self._build_metric_evaluators(report, error_margin)

        rows: list[dict[str, int | float | str]] = []
        for row_index, run in enumerate(runs):
            evaluated = self._evaluate_run(run, threshold, metric_evaluators)
            rows.append(self._build_bisegment_row(run, row_index, evaluated))

        return pd.DataFrame(rows, columns=list(self.BISEGMENTS_COLUMNS))

    def _build_metric_evaluators(
        self,
        report: NoResetClassificationReport[Any, Any],
        error_margin: tuple[int, int],
    ) -> dict[str, Any]:
        """Build single-run metric evaluators used for each bisegment row."""
        report_any = cast(Any, report)
        return {
            "tp": wrap_noreset_single_run_metric(
                source=TruePositiveCount(error_margin),
                policy=cast(Any, self.policy_from_report_base(report_any, "tp")),
            ),
            "fp": wrap_noreset_single_run_metric(
                source=FalsePositiveCount(error_margin),
                policy=cast(Any, self.policy_from_report_base(report_any, "fp")),
            ),
            "fn": wrap_noreset_single_run_metric(
                source=FalseNegativeCount(error_margin),
                policy=cast(Any, self.policy_from_report_base(report_any, "fn")),
            ),
            "precision_tp": wrap_noreset_single_run_metric(
                source=TruePositiveCount(error_margin),
                policy=cast(Any, self.policy_from_report_derived(report_any, "precision", "tp")),
            ),
            "precision_fp": wrap_noreset_single_run_metric(
                source=FalsePositiveCount(error_margin),
                policy=cast(Any, self.policy_from_report_derived(report_any, "precision", "fp")),
            ),
            "recall_tp": wrap_noreset_single_run_metric(
                source=TruePositiveCount(error_margin),
                policy=cast(Any, self.policy_from_report_derived(report_any, "recall", "tp")),
            ),
            "recall_fn": wrap_noreset_single_run_metric(
                source=FalseNegativeCount(error_margin),
                policy=cast(Any, self.policy_from_report_derived(report_any, "recall", "fn")),
            ),
        }

    @staticmethod
    def _evaluate_run(
        run: SingleRun[OnlineDetectionTrace[Any], Any],
        threshold: float,
        metric_evaluators: dict[str, Any],
    ) -> dict[str, int | float]:
        """Evaluate TP, FP, FN, precision, recall, and F1 for one bisegment run."""
        tp_i = int(metric_evaluators["tp"].evaluate(run)(threshold))
        fp_i = int(metric_evaluators["fp"].evaluate(run)(threshold))
        fn_i = int(metric_evaluators["fn"].evaluate(run)(threshold))
        precision_tp_i = int(metric_evaluators["precision_tp"].evaluate(run)(threshold))
        precision_fp_i = int(metric_evaluators["precision_fp"].evaluate(run)(threshold))
        recall_tp_i = int(metric_evaluators["recall_tp"].evaluate(run)(threshold))
        recall_fn_i = int(metric_evaluators["recall_fn"].evaluate(run)(threshold))

        precision = (
            float(precision_tp_i / (precision_tp_i + precision_fp_i)) if (precision_tp_i + precision_fp_i) > 0 else 1.0
        )
        recall = float(recall_tp_i / (recall_tp_i + recall_fn_i)) if (recall_tp_i + recall_fn_i) > 0 else 0.0
        denom = precision + recall
        f1 = 2.0 * precision * recall / denom if denom > 0 else 0.0

        return {
            "tp": tp_i,
            "fp": fp_i,
            "fn": fn_i,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def _build_bisegment_row(
        run: SingleRun[OnlineDetectionTrace[Any], Any],
        row_index: int,
        evaluated: dict[str, int | float],
    ) -> dict[str, int | float | str]:
        """Build one output row for a bisegment run."""
        annotation = cast(BisegmentAnnotation, run.provider.annotation)
        transition_repr = repr(annotation.transition)
        return {
            "bisegment_name": annotation.name,
            "source": annotation.source or "NA",
            "transition": transition_repr,
            "tp": int(evaluated["tp"]),
            "fp": int(evaluated["fp"]),
            "fn": int(evaluated["fn"]),
            "precision": float(evaluated["precision"]),
            "recall": float(evaluated["recall"]),
            "f1": float(evaluated["f1"]),
        }

    @staticmethod
    def extract_error_margin_from_report(
        report: NoResetClassificationReport[Any, Any],
    ) -> tuple[int, int]:
        """Extract the (left, right) error margin from a classification report.

        Traverses the report's ``tp`` source metric to recover the
        ``error_margin`` tuple used by the underlying
        ``TruePositiveCount`` metric.

        Parameters
        ----------
        report
            Report containing the ``tp`` source metric definition.

        Returns
        -------
        tuple[int, int]
            The (left, right) error margin.

        Raises
        ------
        ValueError
            If the tp metric or its error margin cannot be resolved.
        """
        tp_metric = report.bases.get("tp")
        if tp_metric is None:
            raise ValueError("Classification report is missing 'tp' source metric")

        multiple_run_tp = getattr(tp_metric, "source", None)
        single_run_tp = getattr(multiple_run_tp, "base_metric", None) if multiple_run_tp is not None else None
        error_margin = getattr(single_run_tp, "_error_margin", None)

        if (
            not isinstance(error_margin, tuple)
            or len(error_margin) != 2
            or not all(isinstance(value, int) for value in error_margin)
        ):
            raise ValueError("Cannot infer error_margin from classification report")
        return cast(tuple[int, int], error_margin)

    @staticmethod
    def policy_from_report_base(report_for_policy: Any, base_name: str) -> Any:
        """Retrieve the no-reset policy for a source metric from the report.

        Parameters
        ----------
        report_for_policy
            Classification report (duck-typed) containing source metrics.
        base_name
            Key identifying the source metric (e.g. ``'tp'``, ``'fp'``).

        Returns
        -------
        Any
            The policy object attached to the source metric.

        Raises
        ------
        ValueError
            If the source metric or its policy is missing.
        """
        metric = report_for_policy.bases.get(base_name)
        if metric is None:
            raise ValueError(f"Classification report is missing '{base_name}' source metric")
        policy = getattr(metric, "policy", None)
        if policy is None:
            raise ValueError(f"Classification report source '{base_name}' has no no-reset policy")
        return policy

    @staticmethod
    def policy_from_report_derived(
        report_for_policy: Any,
        derived_base_name: str,
        nested_base_name: str,
    ) -> Any:
        """Retrieve the no-reset policy for a derived metric from the report.

        Navigates through a derived metric (e.g. ``precision``) to its
        nested source metrics to locate the policy.

        Parameters
        ----------
        report_for_policy
            Classification report (duck-typed) containing derived metrics.
        derived_base_name
            Key identifying the derived metric (e.g. ``'precision'``).
        nested_base_name
            Key identifying the nested source metric (e.g. ``'tp'``).

        Returns
        -------
        Any
            The policy object attached to the nested metric.

        Raises
        ------
        ValueError
            If the derived metric, nested source, or policy is missing.
        """
        derived_metric = report_for_policy.bases.get(derived_base_name)
        if derived_metric is None:
            raise ValueError(f"Classification report is missing '{derived_base_name}' derived metric")

        nested_bases = getattr(derived_metric, "bases", None)
        if not isinstance(nested_bases, Mapping):
            raise ValueError(f"Classification report derived metric '{derived_base_name}' has invalid bases")

        nested_metric = nested_bases.get(nested_base_name)
        if nested_metric is None:
            raise ValueError(
                f"Classification report derived metric '{derived_base_name}' is missing '{nested_base_name}' source",
            )

        policy = getattr(nested_metric, "policy", None)
        if policy is None:
            raise ValueError(
                f"Classification report derived metric '{derived_base_name}:{nested_base_name}' has no no-reset policy",
            )
        return policy
