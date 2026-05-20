# -*- coding: ascii -*-
"""
Tests for individual bisegment.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from types import SimpleNamespace

import pytest

from pysatl_cpd.analysis.metrics.single_run import TruePositiveCount
from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.individual_bisegment import (
    NoResetBisegmentAnalyzer,
    NoResetBisegmentAnalyzer as Analyzer,
)
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.core.single_run import SingleRun
from tests.support.providers import make_univariate_labeled


class _EvalMetric:
    def __init__(self, value: int) -> None:
        self.value = value

    def evaluate(self, run):
        del run
        return lambda threshold: self.value


def _make_bisegment_run() -> SingleRun:
    provider = make_univariate_labeled(name="series").query_bisegments()[0]
    return SingleRun(
        trace=DetectionTrace(
            detected_change_points=[], detector_description=ChangePointDetectorDescription(name="detector")
        ),
        provider=provider,
    )


def test_analyze_returns_empty_dataframe_for_no_runs(monkeypatch) -> None:
    analyzer = NoResetBisegmentAnalyzer()
    analyzer.classification_report = SimpleNamespace(bases={})
    monkeypatch.setattr(analyzer, "pick_runs", lambda *args, **kwargs: [])

    result = analyzer.analyze(
        SimpleNamespace(description=ChangePointDetectorDescription(name="detector")), threshold=1.0
    )

    assert list(result.columns) == list(analyzer.BISEGMENTS_COLUMNS)
    assert result.empty


def test_evaluate_run_handles_zero_denominator_cases() -> None:
    evaluated = Analyzer._evaluate_run(
        _make_bisegment_run(),
        1.0,
        {
            "tp": _EvalMetric(0),
            "fp": _EvalMetric(0),
            "fn": _EvalMetric(1),
            "precision_tp": _EvalMetric(0),
            "precision_fp": _EvalMetric(0),
            "recall_tp": _EvalMetric(0),
            "recall_fn": _EvalMetric(1),
        },
    )

    assert evaluated == {"tp": 0, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.0, "f1": 0.0}


def test_build_bisegment_row_formats_expected_columns() -> None:
    row = Analyzer._build_bisegment_row(
        _make_bisegment_run(),
        2,
        {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
    )

    assert row["bisegment_name"].endswith("]")
    assert row["source"] == "tests"


def test_extract_error_margin_and_policy_helpers_validate_structure() -> None:
    report = SimpleNamespace(
        bases={
            "tp": SimpleNamespace(source=SimpleNamespace(base_metric=TruePositiveCount((1, 2))), policy="base-policy"),
            "precision": SimpleNamespace(bases={"tp": SimpleNamespace(policy="nested-policy")}),
        }
    )

    assert Analyzer.extract_error_margin_from_report(report) == (1, 2)
    assert Analyzer.policy_from_report_base(report, "tp") == "base-policy"
    assert Analyzer.policy_from_report_derived(report, "precision", "tp") == "nested-policy"

    with pytest.raises(ValueError, match="missing 'fp'"):
        Analyzer.policy_from_report_base(report, "fp")
    with pytest.raises(ValueError, match="missing 'recall'"):
        Analyzer.policy_from_report_derived(report, "recall", "tp")
