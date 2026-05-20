# -*- coding: ascii -*-
"""
Tests for analyzers base.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from types import SimpleNamespace
from typing import Any, cast

import pytest

from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry
from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import ManualThresholdsRange
from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.individual_bisegment import (
    NoResetBisegmentAnalyzer,
)
from pysatl_cpd.benchmark.online.noreset.tooling.pickers import OnlineNoResetEntryAlgorithmPicker
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.data.typedefs import ProviderType
from tests.support.algorithms import CountingAlgorithm, CountingAlgorithmConfig

_Analyzer = NoResetBisegmentAnalyzer


class TestNoResetAnalyzerBase:
    @staticmethod
    def _make_entry() -> OnlineNoResetBenchmarkEntry:
        return OnlineNoResetBenchmarkEntry(
            algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
            thresholds=ManualThresholdsRange(_values=(1.0, 2.0)),
        )

    def test_registry_getter_raises_when_not_set(self) -> None:
        analyzer = _Analyzer()
        with pytest.raises(ValueError, match="Registry is not set"):
            _ = analyzer.registry

    def test_registry_setter_delegates_to_set_registry(self) -> None:
        analyzer = _Analyzer()
        registry = cast(Any, SimpleNamespace())
        analyzer.registry = registry
        assert analyzer._registry is registry

    def test_pick_runs_with_custom_entries_picker_overrides_default(self) -> None:
        analyzer = _Analyzer()
        mock_run = cast(Any, SimpleNamespace(custom=True))
        analyzer._registry = cast(Any, {"k1": mock_run})
        custom_picker = SimpleNamespace(pick=lambda keys, entry: ["k1"])
        result = analyzer.pick_runs(self._make_entry(), entries_picker=custom_picker)
        assert len(result) == 1
        assert cast(Any, result[0]).custom is True

    def test_pick_runs_defaults_to_own_entries_picker(self) -> None:
        analyzer = _Analyzer()
        assert isinstance(analyzer.entries_picker, OnlineNoResetEntryAlgorithmPicker)

    def test_pick_runs_custom_picker_receives_registry_keys_and_entry(self) -> None:
        analyzer = _Analyzer()
        analyzer._registry = cast(Any, {"k1": SimpleNamespace(), "k2": SimpleNamespace()})
        captured_keys = None
        captured_entry = None

        def tracking_picker(keys: Any, entry: Any) -> list[str]:
            nonlocal captured_keys, captured_entry
            captured_keys = list(keys)
            captured_entry = entry
            return []

        expected_entry = self._make_entry()
        analyzer.pick_runs(expected_entry, entries_picker=SimpleNamespace(pick=tracking_picker))
        assert captured_keys == ["k1", "k2"]
        assert captured_entry == expected_entry


class TestNoResetBisigementClassificationMixin:
    def test_classification_report_getter_raises_when_not_set(self) -> None:
        analyzer = _Analyzer()
        with pytest.raises(ValueError, match="Classification report is not set"):
            _ = analyzer.classification_report

    def test_classification_report_setter_delegates(self) -> None:
        analyzer = _Analyzer()
        report = SimpleNamespace()
        analyzer.classification_report = cast(Any, report)
        assert cast(Any, analyzer)._classification_report is report

    @staticmethod
    def _make_run(
        *,
        detector_name: str = "detector",
        provider_type: ProviderType = ProviderType.BISEGMENT,
        change_points: tuple[int, ...] = (3,),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            trace=SimpleNamespace(detector_description=ChangePointDetectorDescription(name=detector_name)),
            provider=SimpleNamespace(
                annotation=SimpleNamespace(provider_type=provider_type),
                change_points=list(change_points),
            ),
        )

    def test_validate_bisegment_runs_detector_mismatch(self) -> None:
        runs = [self._make_run(detector_name="other")]
        expected = ChangePointDetectorDescription(name="expected")
        with pytest.raises(ValueError, match="does not match benchmark entry"):
            _Analyzer.validate_bisegment_runs(cast(Any, runs), expected)

    def test_validate_bisegment_runs_wrong_provider_type(self) -> None:
        runs = [self._make_run(provider_type=ProviderType.NO_CHANGE)]
        expected = ChangePointDetectorDescription(name="detector")
        with pytest.raises(ValueError, match="requires bisegment providers"):
            _Analyzer.validate_bisegment_runs(cast(Any, runs), expected)

    def test_validate_bisegment_runs_wrong_change_points_count(self) -> None:
        runs = [self._make_run(change_points=(3, 5))]
        expected = ChangePointDetectorDescription(name="detector")
        with pytest.raises(ValueError, match="exactly one true change point"):
            _Analyzer.validate_bisegment_runs(cast(Any, runs), expected)

    def test_validate_bisegment_runs_passes_for_valid_run(self) -> None:
        runs = [self._make_run()]
        expected = ChangePointDetectorDescription(name="detector")
        _Analyzer.validate_bisegment_runs(cast(Any, runs), expected)
