# -*- coding: ascii -*-
"""
Tests for state arl extra.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.state_arl import NoResetArlAnalyzer
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.typedefs import ProviderType, StateDescriptor


class TestNoResetArlAnalyzerAnalyze:
    def test_analyze_exercises_picker_and_validation(self) -> None:
        analyzer = NoResetArlAnalyzer()
        analyzer._registry = BenchmarkRegistry()

        state = StateDescriptor(type="baseline")
        entry = MagicMock()
        entry.description = ChangePointDetectorDescription(name="test")

        trace = MagicMock()
        trace.detector_description = ChangePointDetectorDescription(name="test")

        provider = MagicMock()
        provider.__len__ = MagicMock(return_value=100)
        provider.annotation = SimpleNamespace(
            provider_type=ProviderType.NO_CHANGE,
            state=state,
        )

        run = SingleRun(trace=trace, provider=provider)
        analyzer.pick_runs = MagicMock(return_value=[run])
        analyzer.analyze_runs = MagicMock(return_value=pd.DataFrame({"threshold": [0.5], "arl": [50.0]}))

        result = analyzer.analyze(entry, state, thresholds=[0.5], arl_length=100)

        assert isinstance(result, pd.DataFrame)
        analyzer.pick_runs.assert_called_once()
        analyzer.analyze_runs.assert_called_once()

    def test_analyze_runs_returns_dataframe(self) -> None:
        analyzer = NoResetArlAnalyzer()

        trace = MagicMock()
        trace.detector_description = ChangePointDetectorDescription(name="test")

        provider = MagicMock()
        provider.__len__ = MagicMock(return_value=100)
        provider.annotation = SimpleNamespace(
            provider_type=ProviderType.NO_CHANGE,
            state=StateDescriptor(type="baseline"),
        )

        run = SingleRun(trace=trace, provider=provider)

        result = analyzer.analyze_runs([run], thresholds=[0.5, 1.0, 1.5])
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["threshold", "arl"]
        assert len(result) == 3

    def test_validate_arl_runs_passes_for_valid(self) -> None:
        trace = MagicMock()
        trace.detector_description = ChangePointDetectorDescription(name="test")

        provider = MagicMock()
        provider.__len__ = MagicMock(return_value=100)
        provider.annotation = SimpleNamespace(
            provider_type=ProviderType.NO_CHANGE,
            state=StateDescriptor(type="baseline"),
        )

        run = SingleRun(trace=trace, provider=provider)
        NoResetArlAnalyzer.validate_arl_runs(
            [run],
            ChangePointDetectorDescription(name="test"),
            StateDescriptor(type="baseline"),
            arl_length=100,
        )
