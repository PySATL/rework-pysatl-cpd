# -*- coding: ascii -*-
"""
Tests for classification table global.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry
from pysatl_cpd.benchmark.online.noreset.scenarios.classification_table_global import (
    NoResetClassificationTableScenario,
)
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.core.online import OnlineDetectionTrace
from pysatl_cpd.data import Dataset
from pysatl_cpd.data.providers.labeled import PlainUnivariateLabeledData
from pysatl_cpd.data.providers.plain.np_univariate import NDArrayUnivariateProvider
from pysatl_cpd.data.typedefs import SegmentInfo, StateDescriptor, TimeseriesAnnotation, UnlabeledTimeseriesAnnotation
from pysatl_cpd.typedefs import frozendict
from tests.support.algorithms import CountingAlgorithm, CountingAlgorithmConfig


def _make_bisegment_provider(name: str) -> PlainUnivariateLabeledData[TimeseriesAnnotation]:
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    segments = [
        SegmentInfo(segment_num=0, segment_start=0, segment_end=2, state=StateDescriptor(type="a")),
        SegmentInfo(segment_num=1, segment_start=3, segment_end=5, state=StateDescriptor(type="b")),
    ]
    annotation = TimeseriesAnnotation(name=name, metadata=frozendict())
    provider = NDArrayUnivariateProvider(data, UnlabeledTimeseriesAnnotation(name=name))
    return PlainUnivariateLabeledData.from_unlabeled_data(provider, segments, annotation)


class TestNoResetClassificationTableScenario:
    def test_prepare_benchmark_jobs_uses_bisegment_providers(self) -> None:
        dataset = Dataset([_make_bisegment_provider("bisegment-series")])
        scenario = NoResetClassificationTableScenario([])

        jobs = scenario.prepare_benchmark_jobs(dataset)
        assert len(jobs) == 0

    def test_prepare_benchmark_jobs_with_entries(self) -> None:
        provider = _make_bisegment_provider("bisegment-series")
        dataset = Dataset([provider])
        entry = OnlineNoResetBenchmarkEntry(
            algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
            thresholds=MagicMock(),
        )
        scenario = NoResetClassificationTableScenario([entry])

        jobs = scenario.prepare_benchmark_jobs(dataset)
        assert len(jobs) == 1
        assert len(jobs[0].providers) == 1

    def test_analyze_returns_empty_dict_for_no_entries(self) -> None:
        scenario = NoResetClassificationTableScenario([])
        registry: BenchmarkRegistry[Any, OnlineDetectionTrace[Any]] = BenchmarkRegistry()
        scenario.set_classification_report(MagicMock())
        result = scenario.analyze(registry)
        assert result == {}

    def test_set_registry_and_classification_report(self) -> None:
        scenario = NoResetClassificationTableScenario([])
        registry: BenchmarkRegistry[Any, OnlineDetectionTrace[Any]] = BenchmarkRegistry()
        scenario.set_registry(registry)
        assert scenario._classification_analyzer.registry is registry

        report = MagicMock()
        scenario.set_classification_report(report)
        assert scenario._classification_analyzer.classification_report is report

    def test_analyze_with_entries(self) -> None:
        provider = _make_bisegment_provider("bisegment-series")
        Dataset([provider])
        entry = OnlineNoResetBenchmarkEntry(
            algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
            thresholds=MagicMock(),
        )
        scenario = NoResetClassificationTableScenario([entry])
        scenario.set_classification_report(MagicMock())

        scenario._classification_analyzer.pick_runs = MagicMock(return_value=[])
        scenario._threshold_resolver.resolve_classification_thresholds = MagicMock(return_value=[])
        scenario._classification_analyzer.analyze = MagicMock(return_value=pd.DataFrame())

        registry: BenchmarkRegistry[Any, OnlineDetectionTrace[Any]] = BenchmarkRegistry()
        result = scenario.analyze(registry)

        assert entry.description in result
        scenario._classification_analyzer.pick_runs.assert_called_once_with(entry)
        scenario._threshold_resolver.resolve_classification_thresholds.assert_called_once()
        scenario._classification_analyzer.analyze.assert_called_once()
