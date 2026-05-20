# -*- coding: ascii -*-
"""
Tests for individual bisegments table.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pandas as pd
import pytest

from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry
from pysatl_cpd.benchmark.online.noreset.scenarios.individual_bisegments_table import (
    NoResetBisegmentsTableScenario,
)
from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import ManualThresholdsRange
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.benchmark.scenarios import BenchmarkJob
from pysatl_cpd.data import Dataset
from tests.support.algorithms import CountingAlgorithm, CountingAlgorithmConfig
from tests.support.providers import make_univariate_labeled

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(threshold: float = 3.0) -> OnlineNoResetBenchmarkEntry:
    return OnlineNoResetBenchmarkEntry(
        algorithm=CountingAlgorithm(CountingAlgorithmConfig(threshold=threshold)),
        thresholds=ManualThresholdsRange(_values=(1.0, 2.0)),
    )


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_threshold_is_zero(self) -> None:
        scenario = NoResetBisegmentsTableScenario([_make_entry()])
        assert scenario.threshold == 0.0

    def test_custom_threshold(self) -> None:
        scenario = NoResetBisegmentsTableScenario([_make_entry()], threshold=3.5)
        assert scenario.threshold == 3.5

    def test_creates_bisegment_analyzer(self) -> None:
        scenario = NoResetBisegmentsTableScenario([_make_entry()], threshold=0.5)
        assert hasattr(scenario, "_bisegment_analyzer")


# ---------------------------------------------------------------------------
# prepare_benchmark_jobs
# ---------------------------------------------------------------------------


class TestPrepareBenchmarkJobs:
    def test_creates_one_job_per_entry(self) -> None:
        entries = [_make_entry(), _make_entry()]
        scenario = NoResetBisegmentsTableScenario(entries, threshold=0.5)
        dataset = Dataset([make_univariate_labeled(name="series")])
        jobs = scenario.prepare_benchmark_jobs(dataset)
        assert len(jobs) == 2

    def test_job_has_bisegment_providers(self) -> None:
        entry = _make_entry()
        scenario = NoResetBisegmentsTableScenario([entry], threshold=0.5)
        dataset = Dataset([make_univariate_labeled(name="series")])
        jobs = scenario.prepare_benchmark_jobs(dataset)
        assert len(jobs) == 1
        assert len(jobs[0].providers) > 0

    def test_returns_benchmark_job_objects(self) -> None:
        entry = _make_entry()
        scenario = NoResetBisegmentsTableScenario([entry], threshold=0.5)
        dataset = Dataset([make_univariate_labeled(name="series")])
        jobs = scenario.prepare_benchmark_jobs(dataset)
        assert isinstance(jobs[0], BenchmarkJob)

    def test_job_detector_is_not_none(self) -> None:
        entry = _make_entry()
        scenario = NoResetBisegmentsTableScenario([entry], threshold=0.5)
        dataset = Dataset([make_univariate_labeled(name="series")])
        jobs = scenario.prepare_benchmark_jobs(dataset)
        assert jobs[0].detector is not None

    def test_entry_description_matches_factory_detector_description(self) -> None:
        entry = _make_entry()
        scenario = NoResetBisegmentsTableScenario([entry], threshold=0.5)

        expected = entry.description
        detector_description = scenario._make_detector(entry).description

        assert expected == detector_description


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_returns_dict_with_entry_descriptions(self, monkeypatch) -> None:
        entry = _make_entry()
        scenario = NoResetBisegmentsTableScenario([entry], threshold=0.5)

        monkeypatch.setattr(scenario, "set_registry", lambda registry: None)
        monkeypatch.setattr(
            scenario._bisegment_analyzer,
            "analyze",
            lambda entry, threshold: pd.DataFrame({"bisegment_id": ["seg|0"], "precision": [0.9]}),
        )

        registry: BenchmarkRegistry[float, object] = BenchmarkRegistry()
        result = scenario.analyze(registry)

        assert entry.description in result
        assert isinstance(result[entry.description], pd.DataFrame)

    def test_analyze_passes_threshold(self, monkeypatch) -> None:
        entry = _make_entry()
        scenario = NoResetBisegmentsTableScenario([entry], threshold=2.5)
        seen_thresholds: list[float] = []

        monkeypatch.setattr(scenario, "set_registry", lambda registry: None)
        monkeypatch.setattr(
            scenario._bisegment_analyzer,
            "analyze",
            lambda entry, threshold: (
                seen_thresholds.append(threshold) or pd.DataFrame({"bisegment_id": ["seg|0"], "precision": [0.9]})
            ),
        )

        registry: BenchmarkRegistry[float, object] = BenchmarkRegistry()
        scenario.analyze(registry)

        assert seen_thresholds == [2.5]

    def test_analyze_sets_registry_on_analyzer(self, monkeypatch) -> None:
        entry = _make_entry()
        scenario = NoResetBisegmentsTableScenario([entry], threshold=0.5)

        registry: BenchmarkRegistry[float, object] = BenchmarkRegistry()

        monkeypatch.setattr(
            scenario._bisegment_analyzer,
            "analyze",
            lambda entry, threshold: pd.DataFrame({"bisegment_id": ["seg|0"]}),
        )

        scenario.analyze(registry)
        assert scenario._bisegment_analyzer.registry is registry

    def test_analyze_multiple_entries(self, monkeypatch) -> None:
        entry_a = _make_entry(threshold=3.0)
        entry_b = _make_entry(threshold=5.0)
        scenario = NoResetBisegmentsTableScenario([entry_a, entry_b], threshold=0.5)

        monkeypatch.setattr(scenario, "set_registry", lambda registry: None)
        monkeypatch.setattr(
            scenario._bisegment_analyzer,
            "analyze",
            lambda entry, threshold: pd.DataFrame({"bisegment_id": ["seg|0"], "precision": [0.9]}),
        )

        registry: BenchmarkRegistry[float, object] = BenchmarkRegistry()
        result = scenario.analyze(registry)

        assert len(result) == 2
        assert entry_a.description in result
        assert entry_b.description in result


# ---------------------------------------------------------------------------
# set_registry / set_classification_report
# ---------------------------------------------------------------------------


class TestSetters:
    def test_set_registry_forwards_to_analyzer(self) -> None:
        scenario = NoResetBisegmentsTableScenario([_make_entry()], threshold=0.5)
        registry: BenchmarkRegistry[float, object] = BenchmarkRegistry()
        scenario.set_registry(registry)
        assert scenario._bisegment_analyzer.registry is registry

    def test_set_classification_report_forwards_to_analyzer(self) -> None:
        scenario = NoResetBisegmentsTableScenario([_make_entry()], threshold=0.5)
        report = object()
        scenario.set_classification_report(report)
        assert scenario._bisegment_analyzer.classification_report is report


# ---------------------------------------------------------------------------
# handle_benchmark_error
# ---------------------------------------------------------------------------


class TestHandleBenchmarkError:
    def test_raises_value_error_with_data_shape_message(self) -> None:
        entry = _make_entry()
        scenario = NoResetBisegmentsTableScenario([entry], threshold=0.5)
        dataset = Dataset([make_univariate_labeled(name="series")])
        jobs = scenario.prepare_benchmark_jobs(dataset)
        job = jobs[0]

        with pytest.raises(ValueError) as excinfo:
            scenario.handle_benchmark_error(job, ValueError("original error"))

        msg = str(excinfo.value)
        assert "data shape mismatch" in msg
        assert "data_transformer" in msg
        assert str(job.detector.description) in msg

    def test_chains_original_exception(self) -> None:
        entry = _make_entry()
        scenario = NoResetBisegmentsTableScenario([entry], threshold=0.5)
        dataset = Dataset([make_univariate_labeled(name="series")])
        jobs = scenario.prepare_benchmark_jobs(dataset)
        job = jobs[0]
        original = ValueError("original error")

        with pytest.raises(ValueError) as excinfo:
            scenario.handle_benchmark_error(job, original)

        assert excinfo.value.__cause__ is original
