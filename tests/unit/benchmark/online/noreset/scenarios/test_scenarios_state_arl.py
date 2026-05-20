# -*- coding: ascii -*-
"""
Tests for scenarios state arl.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pandas as pd
import pytest

from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry
from pysatl_cpd.benchmark.online.noreset.scenarios.state_arl import NoResetArlByStateScenario
from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import ManualThresholdsRange
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.data import Dataset
from pysatl_cpd.data.typedefs import StateDescriptor
from tests.support.algorithms import CountingAlgorithm, CountingAlgorithmConfig
from tests.support.providers import make_univariate_labeled


def _make_entry() -> OnlineNoResetBenchmarkEntry:
    return OnlineNoResetBenchmarkEntry(
        algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
        thresholds=ManualThresholdsRange(_values=(1.0, 2.0)),
    )


def test_prepare_benchmark_jobs_rejects_missing_state_segments() -> None:
    scenario = NoResetArlByStateScenario([_make_entry()], state=StateDescriptor(type="missing"), arl_length=4)
    dataset = Dataset([make_univariate_labeled(name="series")])

    with pytest.raises(ValueError, match="No segments in state"):
        scenario.prepare_benchmark_jobs(dataset)


def test_prepare_benchmark_jobs_skips_arl_when_merged_length_is_too_short() -> None:
    scenario = NoResetArlByStateScenario([_make_entry()], state=StateDescriptor(label="baseline"), arl_length=50)
    dataset = Dataset([make_univariate_labeled(name="series")])

    jobs = scenario.prepare_benchmark_jobs(dataset)

    assert scenario._has_arl_providers is False
    assert len(jobs) == 1
    assert jobs[0].providers == []


def test_analyze_returns_empty_without_arl_providers() -> None:
    scenario = NoResetArlByStateScenario([_make_entry()], state=StateDescriptor(label="baseline"), arl_length=5)
    scenario._has_arl_providers = False

    assert scenario.analyze(BenchmarkRegistry()) == {}


def test_prepare_benchmark_jobs_with_arl_data_creates_jobs() -> None:
    """When StateDataset.from_dataset returns data, jobs include ARL providers."""
    scenario = NoResetArlByStateScenario([_make_entry()], state=StateDescriptor(label="baseline"), arl_length=3)
    dataset = Dataset([make_univariate_labeled(name="series")])

    jobs = scenario.prepare_benchmark_jobs(dataset)

    assert scenario._has_arl_providers is True
    assert len(jobs) == 1
    assert len(jobs[0].providers) > 0


def test_prepare_benchmark_jobs_with_arl_data_multiple_entries() -> None:
    """Multiple entries get separate jobs with ARL providers."""
    scenario = NoResetArlByStateScenario(
        [_make_entry(), _make_entry()], state=StateDescriptor(label="baseline"), arl_length=3
    )
    dataset = Dataset([make_univariate_labeled(name="series")])

    jobs = scenario.prepare_benchmark_jobs(dataset)

    assert scenario._has_arl_providers is True
    assert len(jobs) == 2


def test_analyze_filters_runs_by_length_and_uses_resolver(monkeypatch) -> None:
    entry = _make_entry()
    scenario = NoResetArlByStateScenario([entry], state=StateDescriptor(label="baseline"), arl_length=4)
    scenario._has_arl_providers = True
    registry = BenchmarkRegistry()

    matching = type("Run", (), {"provider": type("Provider", (), {"__len__": lambda self: 4})()})()
    short = type("Run", (), {"provider": type("Provider", (), {"__len__": lambda self: 3})()})()

    monkeypatch.setattr(scenario._arl_analyzer, "pick_runs", lambda *args, **kwargs: [matching, short])
    monkeypatch.setattr(scenario._arl_analyzer, "validate_arl_runs", lambda runs, desc, state, length: None)
    monkeypatch.setattr(
        scenario._threshold_resolver, "resolve_arl_thresholds", lambda entry, runs, thresholds: [0.5, 1.5]
    )
    monkeypatch.setattr(
        scenario._arl_analyzer,
        "analyze_runs",
        lambda runs, thresholds: pd.DataFrame({"threshold": thresholds, "arl": [10.0, 20.0]}),
    )

    result = scenario.analyze(registry)

    assert list(result[entry.description]["threshold"]) == [0.5, 1.5]
    assert list(result[entry.description]["arl"]) == [10.0, 20.0]
