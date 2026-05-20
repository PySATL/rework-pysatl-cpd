# -*- coding: ascii -*-
"""
Tests for individual bisegment error paths.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from types import SimpleNamespace

import pytest

from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.individual_bisegment import NoResetBisegmentAnalyzer

_Analyzer = NoResetBisegmentAnalyzer


class TestExtractErrorMarginFromReport:
    def test_missing_tp_raises(self) -> None:
        report = SimpleNamespace(bases={})
        with pytest.raises(ValueError, match="missing 'tp'"):
            _Analyzer.extract_error_margin_from_report(report)

    def test_invalid_error_margin_not_a_tuple_raises(self) -> None:
        tp_metric = SimpleNamespace(source=SimpleNamespace(base_metric=SimpleNamespace(_error_margin="not-a-tuple")))
        report = SimpleNamespace(bases={"tp": tp_metric})
        with pytest.raises(ValueError, match="Cannot infer error_margin"):
            _Analyzer.extract_error_margin_from_report(report)

    def test_invalid_error_margin_wrong_length_raises(self) -> None:
        tp_metric = SimpleNamespace(source=SimpleNamespace(base_metric=SimpleNamespace(_error_margin=(1, 2, 3))))
        report = SimpleNamespace(bases={"tp": tp_metric})
        with pytest.raises(ValueError, match="Cannot infer error_margin"):
            _Analyzer.extract_error_margin_from_report(report)

    def test_invalid_error_margin_non_int_raises(self) -> None:
        tp_metric = SimpleNamespace(source=SimpleNamespace(base_metric=SimpleNamespace(_error_margin=(1, "x"))))
        report = SimpleNamespace(bases={"tp": tp_metric})
        with pytest.raises(ValueError, match="Cannot infer error_margin"):
            _Analyzer.extract_error_margin_from_report(report)

    def test_extracts_valid_error_margin(self) -> None:
        tp_metric = SimpleNamespace(source=SimpleNamespace(base_metric=SimpleNamespace(_error_margin=(1, 2))))
        report = SimpleNamespace(bases={"tp": tp_metric})
        result = _Analyzer.extract_error_margin_from_report(report)
        assert result == (1, 2)


class TestPolicyFromReportBase:
    def test_missing_base_metric_raises(self) -> None:
        report = SimpleNamespace(bases={})
        with pytest.raises(ValueError, match="missing 'tp'"):
            _Analyzer.policy_from_report_base(report, "tp")

    def test_missing_policy_raises(self) -> None:
        report = SimpleNamespace(bases={"tp": SimpleNamespace(policy=None)})
        with pytest.raises(ValueError, match="no no-reset policy"):
            _Analyzer.policy_from_report_base(report, "tp")

    def test_valid_policy_returns(self) -> None:
        policy = object()
        report = SimpleNamespace(bases={"tp": SimpleNamespace(policy=policy)})
        result = _Analyzer.policy_from_report_base(report, "tp")
        assert result is policy


class TestPolicyFromReportDerived:
    def test_missing_derived_metric_raises(self) -> None:
        report = SimpleNamespace(bases={})
        with pytest.raises(ValueError, match="missing 'precision'"):
            _Analyzer.policy_from_report_derived(report, "precision", "tp")

    def test_invalid_bases_not_mapping_raises(self) -> None:
        derived = SimpleNamespace(bases=None)
        report = SimpleNamespace(bases={"precision": derived})
        with pytest.raises(ValueError, match="invalid bases"):
            _Analyzer.policy_from_report_derived(report, "precision", "tp")

    def test_missing_nested_base_raises(self) -> None:
        derived = SimpleNamespace(bases={})
        report = SimpleNamespace(bases={"precision": derived})
        with pytest.raises(ValueError, match="missing 'tp' source"):
            _Analyzer.policy_from_report_derived(report, "precision", "tp")

    def test_missing_nested_policy_raises(self) -> None:
        derived = SimpleNamespace(bases={"tp": SimpleNamespace(policy=None)})
        report = SimpleNamespace(bases={"precision": derived})
        with pytest.raises(ValueError, match="no no-reset policy"):
            _Analyzer.policy_from_report_derived(report, "precision", "tp")

    def test_valid_derived_policy_returns(self) -> None:
        policy = object()
        derived = SimpleNamespace(bases={"tp": SimpleNamespace(policy=policy)})
        report = SimpleNamespace(bases={"precision": derived})
        result = _Analyzer.policy_from_report_derived(report, "precision", "tp")
        assert result is policy
