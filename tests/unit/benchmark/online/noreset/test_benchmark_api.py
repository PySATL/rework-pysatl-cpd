# -*- coding: ascii -*-
"""
Tests for benchmark api.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import pandas as pd
from _pytest.monkeypatch import MonkeyPatch

from pysatl_cpd.benchmark.online.noreset.benchmark import OnlineNoResetBenchmark
from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry
from pysatl_cpd.benchmark.online.noreset.policy_kind import NoResetPolicyKind
from pysatl_cpd.benchmark.online.noreset.scenarios import (
    NoResetArlByStateScenario,
    NoResetBisegmentsTableScenario,
    NoResetClassificationTableByTransitionScenario,
    NoResetClassificationTableScenario,
)
from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import ManualThresholdsRange
from pysatl_cpd.benchmark.online.noreset.tooling.bisegment_cut import BisegmentCut
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.data import Dataset
from pysatl_cpd.data.typedefs import StateDescriptor, TransitionDescriptor
from tests.support.algorithms import CountingAlgorithm, CountingAlgorithmConfig
from tests.support.providers import make_univariate_labeled


def _make_benchmark() -> tuple[OnlineNoResetBenchmark, OnlineNoResetBenchmarkEntry]:
    dataset = Dataset([make_univariate_labeled(name="series-a")])
    benchmark: OnlineNoResetBenchmark = OnlineNoResetBenchmark(
        dataset,
        BenchmarkRegistry(),
        max_delay=0,
        global_policy=NoResetPolicyKind.POINT,
    )
    entry = OnlineNoResetBenchmarkEntry(
        algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
        thresholds=ManualThresholdsRange(_values=(1.0, 2.0)),
    )
    return benchmark, entry


def test_benchmark_convenience_methods_build_expected_scenarios(monkeypatch: MonkeyPatch) -> None:
    benchmark, entry = _make_benchmark()
    captured: list[tuple[Any, dict[str, Any]]] = []

    def fake_run_scenario(self: OnlineNoResetBenchmark, scenario: Any, **kwargs: Any) -> dict[Any, pd.DataFrame]:
        captured.append((scenario, kwargs))
        return {entry.description: pd.DataFrame()}

    monkeypatch.setattr(OnlineNoResetBenchmark, "run_scenario", fake_run_scenario)

    benchmark.get_classification_table(
        [entry],
        collect_states=True,
        n_jobs=2,
        backend="threading",
    )
    benchmark.get_classification_table_by_transition(
        [entry],
        TransitionDescriptor(curr_state=StateDescriptor(type="baseline"), next_state=StateDescriptor(type="shift")),
        use_arl=True,
        arl_length=10,
    )
    benchmark.get_ARL_table_by_state([entry], StateDescriptor(type="baseline"), arl_length=5)
    benchmark.get_bisegments_table([entry], threshold=2.0)

    assert isinstance(captured[0][0], NoResetClassificationTableScenario)
    assert captured[0][1] == {"n_jobs": 2, "backend": "threading"}
    assert not hasattr(captured[0][0], "bisegment_cut")
    assert isinstance(captured[1][0], NoResetClassificationTableByTransitionScenario)
    assert not hasattr(captured[1][0], "bisegment_cut")
    assert isinstance(captured[2][0], NoResetArlByStateScenario)
    assert not hasattr(captured[2][0], "bisegment_cut")
    assert isinstance(captured[3][0], NoResetBisegmentsTableScenario)
    assert not hasattr(captured[3][0], "bisegment_cut")


def test_benchmark_convenience_methods_propagate_classification_report(monkeypatch: MonkeyPatch) -> None:
    benchmark, entry = _make_benchmark()
    report = benchmark._classification_report
    seen = []

    def fake_run_scenario(self: OnlineNoResetBenchmark, scenario: Any, **kwargs: Any) -> dict[Any, pd.DataFrame]:
        analyzer = getattr(scenario, "_classification_analyzer", getattr(scenario, "_bisegment_analyzer", None))
        seen.append(getattr(analyzer, "_classification_report", None))
        return {entry.description: pd.DataFrame()}

    monkeypatch.setattr(OnlineNoResetBenchmark, "run_scenario", fake_run_scenario)

    benchmark.get_classification_table([entry])
    benchmark.get_classification_table_by_transition(
        [entry],
        TransitionDescriptor(curr_state=StateDescriptor(type="baseline"), next_state=StateDescriptor(type="shift")),
        use_arl=False,
    )
    benchmark.get_ARL_table_by_state([entry], StateDescriptor(type="baseline"), arl_length=5)
    benchmark.get_bisegments_table([entry], threshold=1.0)

    assert seen[0] is report
    assert seen[1] is report
    assert seen[2] is None
    assert seen[3] is report


def test_build_classification_report_policy_kinds() -> None:
    from pysatl_cpd.benchmark.online.noreset.metrics import NoResetClassificationReport
    from pysatl_cpd.benchmark.online.noreset.metrics.policy.bisegment import MixedPolicy, PointBasedPolicy

    built = OnlineNoResetBenchmark.build_classification_report(
        max_delay=7,
        global_policy=NoResetPolicyKind.MIXED,
        precision_policy=NoResetPolicyKind.POINT,
    )
    assert isinstance(built, NoResetClassificationReport)
    assert isinstance(built.bases["tp"].policy, MixedPolicy)
    assert isinstance(built.bases["precision"].bases["tp"].policy, PointBasedPolicy)
    assert isinstance(built.bases["recall"].bases["tp"].policy, MixedPolicy)


def test_get_pr_auc_table_handles_empty_missing_columns_and_duplicates() -> None:
    desc = OnlineNoResetBenchmarkEntry(
        algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
        thresholds=ManualThresholdsRange(_values=(1.0,)),
    ).description

    assert OnlineNoResetBenchmark.get_pr_auc_table({desc: pd.DataFrame()}) == {}

    try:
        OnlineNoResetBenchmark.get_pr_auc_table({desc: pd.DataFrame({"precision": [1.0]})})
    except ValueError as exc:
        assert "recall" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing recall column")

    table = pd.DataFrame(
        {
            "recall": [0.5, 0.5, 1.0],
            "precision": [0.2, 0.8, 0.1],
        }
    )
    result = OnlineNoResetBenchmark.get_pr_auc_table({desc: table})
    assert result[desc] == 0.675


def test_entry_description_changes_with_bisegment_cut() -> None:
    base_entry = OnlineNoResetBenchmarkEntry(
        algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
        thresholds=ManualThresholdsRange(_values=(1.0,)),
    )
    trimmed_entry = OnlineNoResetBenchmarkEntry(
        algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
        thresholds=ManualThresholdsRange(_values=(1.0,)),
        bisegment_cut=BisegmentCut(left_trim=1, right_trim=2),
    )

    assert base_entry.description != trimmed_entry.description


def test_arl_scenario_entry_for_arl_forces_noop_cut() -> None:
    _, entry = _make_benchmark()
    entry = OnlineNoResetBenchmarkEntry(
        algorithm=entry.algorithm,
        thresholds=entry.thresholds,
        bisegment_cut=BisegmentCut(left_trim=3, right_trim=4),
    )
    scenario: NoResetArlByStateScenario[Any] = NoResetArlByStateScenario(
        [entry], state=StateDescriptor(type="baseline"), arl_length=5
    )

    arl_entry = scenario._entry_for_arl(entry)
    assert arl_entry.bisegment_cut.is_noop
    assert arl_entry.algorithm == entry.algorithm
    assert arl_entry.thresholds == entry.thresholds


def test_transition_scenario_entry_for_arl_forces_noop_cut() -> None:
    _, entry = _make_benchmark()
    entry = OnlineNoResetBenchmarkEntry(
        algorithm=entry.algorithm,
        thresholds=entry.thresholds,
        bisegment_cut=BisegmentCut(left_trim=2, right_trim=1),
    )
    transition = TransitionDescriptor(
        curr_state=StateDescriptor(type="baseline"), next_state=StateDescriptor(type="shift")
    )
    scenario: NoResetClassificationTableByTransitionScenario[Any] = NoResetClassificationTableByTransitionScenario(
        [entry], transition=transition, use_arl=True, arl_length=5
    )

    arl_entry = scenario._entry_for_arl(entry)
    assert arl_entry.bisegment_cut.is_noop
    assert arl_entry.algorithm == entry.algorithm
    assert arl_entry.thresholds == entry.thresholds
