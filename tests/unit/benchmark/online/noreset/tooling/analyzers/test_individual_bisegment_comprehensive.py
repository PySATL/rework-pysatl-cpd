# -*- coding: ascii -*-
"""
Tests for individual bisegment comprehensive.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from types import SimpleNamespace

import pytest

from pysatl_cpd.analysis.metrics.single_run import TruePositiveCount
from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.individual_bisegment import (
    NoResetBisegmentAnalyzer as Analyzer,
)
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription


class TestExtractErrorMarginFromReport:
    def test_missing_tp_metric(self) -> None:
        report = SimpleNamespace(bases={})
        with pytest.raises(ValueError, match="missing 'tp' source metric"):
            Analyzer.extract_error_margin_from_report(report)

    def test_error_margin_not_a_tuple(self) -> None:
        report = SimpleNamespace(
            bases={
                "tp": SimpleNamespace(
                    source=SimpleNamespace(base_metric=SimpleNamespace(_error_margin=None)),
                ),
            },
        )
        with pytest.raises(ValueError, match="Cannot infer error_margin"):
            Analyzer.extract_error_margin_from_report(report)

    def test_error_margin_wrong_length(self) -> None:
        report = SimpleNamespace(
            bases={
                "tp": SimpleNamespace(
                    source=SimpleNamespace(base_metric=SimpleNamespace(_error_margin=(1,))),
                ),
            },
        )
        with pytest.raises(ValueError, match="Cannot infer error_margin"):
            Analyzer.extract_error_margin_from_report(report)

    def test_error_margin_non_int_values(self) -> None:
        report = SimpleNamespace(
            bases={
                "tp": SimpleNamespace(
                    source=SimpleNamespace(base_metric=SimpleNamespace(_error_margin=("a", "b"))),
                ),
            },
        )
        with pytest.raises(ValueError, match="Cannot infer error_margin"):
            Analyzer.extract_error_margin_from_report(report)


class TestPolicyFromReportBase:
    def test_missing_metric(self) -> None:
        report = SimpleNamespace(bases={})
        with pytest.raises(ValueError, match="missing 'tp' source metric"):
            Analyzer.policy_from_report_base(report, "tp")

    def test_metric_has_no_policy(self) -> None:
        report = SimpleNamespace(bases={"tp": SimpleNamespace(policy=None)})
        with pytest.raises(ValueError, match="has no no-reset policy"):
            Analyzer.policy_from_report_base(report, "tp")


class TestPolicyFromReportDerived:
    def test_missing_derived_metric(self) -> None:
        report = SimpleNamespace(bases={})
        with pytest.raises(ValueError, match="missing 'precision' derived metric"):
            Analyzer.policy_from_report_derived(report, "precision", "tp")

    def test_invalid_bases_not_a_mapping(self) -> None:
        report = SimpleNamespace(
            bases={"precision": SimpleNamespace(bases=None)},
        )
        with pytest.raises(ValueError, match="has invalid bases"):
            Analyzer.policy_from_report_derived(report, "precision", "tp")

    def test_missing_nested_base(self) -> None:
        report = SimpleNamespace(
            bases={"precision": SimpleNamespace(bases={})},
        )
        with pytest.raises(ValueError, match="missing 'tp' source"):
            Analyzer.policy_from_report_derived(report, "precision", "tp")

    def test_nested_metric_has_no_policy(self) -> None:
        report = SimpleNamespace(
            bases={
                "precision": SimpleNamespace(
                    bases={"tp": SimpleNamespace(policy=None)},
                ),
            },
        )
        with pytest.raises(ValueError, match="has no no-reset policy"):
            Analyzer.policy_from_report_derived(report, "precision", "tp")


class TestBuildMetricEvaluators:
    def test_returns_correct_dict_structure(self) -> None:
        analyzer = Analyzer()
        report = SimpleNamespace(
            bases={
                "tp": SimpleNamespace(
                    source=SimpleNamespace(base_metric=TruePositiveCount((1, 2))),
                    policy="tp-policy",
                ),
                "fp": SimpleNamespace(
                    source=SimpleNamespace(base_metric=TruePositiveCount((1, 2))),
                    policy="fp-policy",
                ),
                "fn": SimpleNamespace(
                    source=SimpleNamespace(base_metric=TruePositiveCount((1, 2))),
                    policy="fn-policy",
                ),
                "precision": SimpleNamespace(
                    bases={
                        "tp": SimpleNamespace(policy="prec-tp-policy"),
                        "fp": SimpleNamespace(policy="prec-fp-policy"),
                    },
                ),
                "recall": SimpleNamespace(
                    bases={
                        "tp": SimpleNamespace(policy="rec-tp-policy"),
                        "fn": SimpleNamespace(policy="rec-fn-policy"),
                    },
                ),
            },
        )
        error_margin = (1, 2)
        result = analyzer._build_metric_evaluators(report, error_margin)

        expected_keys = {"tp", "fp", "fn", "precision_tp", "precision_fp", "recall_tp", "recall_fn"}
        assert set(result.keys()) == expected_keys
        for key in expected_keys:
            assert hasattr(result[key], "evaluate")


class TestAnalyzeWithRuns:
    def test_analyze_calls_validate_bisegment_runs_and_evaluation(self, monkeypatch) -> None:
        analyzer = Analyzer()
        analyzer._classification_report = SimpleNamespace(bases={})
        mock_run = _make_mock_run()

        def fake_pick_runs(entry, *, entries_picker=None):
            return [mock_run]

        monkeypatch.setattr(analyzer, "pick_runs", fake_pick_runs)

        validate_called = False

        def tracking_validate(runs, expected_detector):
            nonlocal validate_called
            validate_called = True
            assert len(runs) == 1

        monkeypatch.setattr(Analyzer, "validate_bisegment_runs", staticmethod(tracking_validate))
        monkeypatch.setattr(Analyzer, "extract_error_margin_from_report", staticmethod(lambda report: (0, 0)))
        monkeypatch.setattr(Analyzer, "_build_metric_evaluators", lambda self, report, margin: {"tp": _EvalMetric(1)})
        monkeypatch.setattr(Analyzer, "_evaluate_run", staticmethod(lambda run, threshold, evaluators: {"tp": 1}))
        monkeypatch.setattr(
            Analyzer,
            "_build_bisegment_row",
            staticmethod(lambda run, index, evaluated: {"bisegment_name": f"row_{index}", "tp": evaluated["tp"]}),
        )

        result = analyzer.analyze(
            SimpleNamespace(description=ChangePointDetectorDescription(name="detector")),
            threshold=1.0,
        )

        assert validate_called
        assert len(result) == 1
        assert result.iloc[0]["bisegment_name"] == "row_0"


def _make_mock_run() -> SimpleNamespace:
    return SimpleNamespace(
        trace=SimpleNamespace(detector_description=ChangePointDetectorDescription(name="detector")),
        provider=SimpleNamespace(
            annotation=SimpleNamespace(
                provider_type="bisegment",
                transition=SimpleNamespace(),
                name="series[0:5](baseline->shift)",
            ),
            change_points=[3],
        ),
    )


class _EvalMetric:
    def __init__(self, value: int) -> None:
        self.value = value

    def evaluate(self, run):
        return lambda threshold: self.value
