# -*- coding: ascii -*-
"""
Tests for classification table.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from pysatl_cpd.benchmark.online.noreset.metrics import NoResetClassificationReport
from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.classification_table import (
    NoResetClassificationTableAnalyzer,
)
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.typedefs import ProviderType


class TestNoResetClassificationTableAnalyzer:
    def test_analyze_returns_empty_dataframe_when_no_runs(self) -> None:
        analyzer = NoResetClassificationTableAnalyzer()
        analyzer._registry = BenchmarkRegistry()
        analyzer._classification_report = MagicMock()

        entry = MagicMock()
        entry.description = ChangePointDetectorDescription(name="test")
        analyzer.pick_runs = MagicMock(return_value=[])

        result = analyzer.analyze(entry, thresholds=[0.5, 1.0])
        assert result.empty
        assert list(result.columns) == [
            "threshold",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "mean_delay",
            "median_delay",
        ]

    def test_set_registry(self) -> None:
        analyzer = NoResetClassificationTableAnalyzer()
        registry: BenchmarkRegistry = BenchmarkRegistry()
        analyzer._set_registry(registry)
        assert analyzer._registry is registry

    def test_set_classification_report(self) -> None:
        analyzer = NoResetClassificationTableAnalyzer()
        report = MagicMock(spec=NoResetClassificationReport)
        analyzer._set_classification_report(report)
        assert analyzer._classification_report is report

    def test_analyze_with_runs(self) -> None:
        from pysatl_cpd.benchmark.online.noreset.detector.noreset_trace import NoResetDetectionTrace

        analyzer = NoResetClassificationTableAnalyzer()
        registry: BenchmarkRegistry = BenchmarkRegistry()
        analyzer._registry = registry

        provider = MagicMock()
        provider.annotation = SimpleNamespace(provider_type=ProviderType.BISEGMENT)
        provider.change_points = (2,)
        detector_desc = ChangePointDetectorDescription(name="test")
        trace = NoResetDetectionTrace(
            detector_description=detector_desc,
            detection_function=np.array([1.0, 2.0, 3.0]),
            processing_time=np.array([], dtype=np.float64),
            algorithm_states=[],
        )
        run = SingleRun(trace=trace, provider=provider)

        report = MagicMock(spec=NoResetClassificationReport)
        report.evaluate.return_value = lambda th: {
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
        tp_base = MagicMock()
        tp_base.source.base_metric._error_margin = (0, 5)
        report.bases = {"tp": tp_base}
        analyzer._classification_report = report

        entry = MagicMock()
        entry.description = detector_desc
        analyzer.pick_runs = MagicMock(return_value=[run])

        result = analyzer.analyze(entry, thresholds=[0.5])
        assert not result.empty
        assert 0.5 in result["threshold"].values
