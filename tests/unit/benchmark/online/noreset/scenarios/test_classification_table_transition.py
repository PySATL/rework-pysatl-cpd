"""
Tests for classification table transition.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pandas as pd
import pytest

from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry
from pysatl_cpd.benchmark.online.noreset.scenarios.classification_table_transition import (
    NoResetClassificationTableByTransitionScenario,
)
from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import ManualThresholdsRange
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.data import Dataset, StateDataset
from pysatl_cpd.data.typedefs import BisegmentInfo, SegmentInfo, StateDescriptor, TransitionDescriptor
from tests.support.algorithms import CountingAlgorithm, CountingAlgorithmConfig
from tests.support.providers import make_univariate_labeled


def _make_entry(threshold: float = 3.0) -> OnlineNoResetBenchmarkEntry:
    return OnlineNoResetBenchmarkEntry(
        algorithm=CountingAlgorithm(CountingAlgorithmConfig(threshold=threshold)),
        thresholds=ManualThresholdsRange(_values=(1.0, 2.0)),
    )


def _make_transition() -> TransitionDescriptor:
    return TransitionDescriptor(
        curr_state=StateDescriptor(label="baseline"),
        next_state=StateDescriptor(label="shift"),
    )


def _other_transition() -> TransitionDescriptor:
    return TransitionDescriptor(
        curr_state=StateDescriptor(label="foo"),
        next_state=StateDescriptor(label="bar"),
    )


# ---------------------------------------------------------------------------
# __init__  -  value errors
# ---------------------------------------------------------------------------


class TestInit:
    def test_rejects_none_transition(self) -> None:
        with pytest.raises(ValueError, match="transition is required"):
            NoResetClassificationTableByTransitionScenario(
                [_make_entry()],
                transition=None,
            )

    def test_rejects_use_arl_without_arl_length(self) -> None:
        with pytest.raises(ValueError, match="use_arl=True requires a positive arl_length"):
            NoResetClassificationTableByTransitionScenario(
                [_make_entry()],
                transition=_make_transition(),
                use_arl=True,
                arl_length=None,
            )

    def test_rejects_use_arl_with_zero_arl_length(self) -> None:
        with pytest.raises(ValueError, match="use_arl=True requires a positive arl_length"):
            NoResetClassificationTableByTransitionScenario(
                [_make_entry()],
                transition=_make_transition(),
                use_arl=True,
                arl_length=0,
            )

    def test_rejects_use_arl_with_negative_arl_length(self) -> None:
        with pytest.raises(ValueError, match="use_arl=True requires a positive arl_length"):
            NoResetClassificationTableByTransitionScenario(
                [_make_entry()],
                transition=_make_transition(),
                use_arl=True,
                arl_length=-5,
            )

    def test_accepts_use_arl_false_without_arl_length(self) -> None:
        scenario = NoResetClassificationTableByTransitionScenario(
            [_make_entry()],
            transition=_make_transition(),
            use_arl=False,
        )
        assert scenario.use_arl is False
        assert scenario.arl_length is None

    def test_accepts_valid_params(self) -> None:
        scenario = NoResetClassificationTableByTransitionScenario(
            [_make_entry()],
            transition=_make_transition(),
            use_arl=True,
            arl_length=100,
        )
        assert scenario.use_arl is True
        assert scenario.arl_length == 100
        assert scenario.transition == _make_transition()


# ---------------------------------------------------------------------------
# transition_checked / arl_length_checked properties
# ---------------------------------------------------------------------------


class TestCheckedProperties:
    def test_transition_checked_raises_when_none(self) -> None:
        scenario = NoResetClassificationTableByTransitionScenario(
            [_make_entry()],
            transition=_make_transition(),
        )
        scenario.transition = None
        with pytest.raises(ValueError, match="transition is required"):
            _ = scenario.transition_checked

    def test_transition_checked_returns_value(self) -> None:
        transition = _make_transition()
        scenario = NoResetClassificationTableByTransitionScenario(
            [_make_entry()],
            transition=transition,
        )
        assert scenario.transition_checked == transition

    def test_arl_length_checked_raises_when_none(self) -> None:
        scenario = NoResetClassificationTableByTransitionScenario(
            [_make_entry()],
            transition=_make_transition(),
            use_arl=True,
            arl_length=100,
        )
        scenario.arl_length = None
        with pytest.raises(ValueError, match="use_arl=True requires a positive arl_length"):
            _ = scenario.arl_length_checked

    def test_arl_length_checked_raises_when_zero(self) -> None:
        scenario = NoResetClassificationTableByTransitionScenario(
            [_make_entry()],
            transition=_make_transition(),
            use_arl=True,
            arl_length=100,
        )
        scenario.arl_length = 0
        with pytest.raises(ValueError, match="use_arl=True requires a positive arl_length"):
            _ = scenario.arl_length_checked

    def test_arl_length_checked_raises_when_negative(self) -> None:
        scenario = NoResetClassificationTableByTransitionScenario(
            [_make_entry()],
            transition=_make_transition(),
            use_arl=True,
            arl_length=100,
        )
        scenario.arl_length = -1
        with pytest.raises(ValueError, match="use_arl=True requires a positive arl_length"):
            _ = scenario.arl_length_checked

    def test_arl_length_checked_returns_value(self) -> None:
        scenario = NoResetClassificationTableByTransitionScenario(
            [_make_entry()],
            transition=_make_transition(),
            use_arl=True,
            arl_length=50,
        )
        assert scenario.arl_length_checked == 50


# ---------------------------------------------------------------------------
# Static helper methods  (_get_segments_filter_by_state,
#                         _get_bisegments_filter_by_transition)
# ---------------------------------------------------------------------------


class TestStaticFilters:
    def test_get_segments_filter_by_state_matches(self) -> None:
        filter_fn = NoResetClassificationTableByTransitionScenario._get_segments_filter_by_state(
            StateDescriptor(label="baseline")
        )
        matching = SegmentInfo(segment_num=0, segment_start=0, segment_end=2, state=StateDescriptor(label="baseline"))
        non_matching = SegmentInfo(segment_num=1, segment_start=3, segment_end=5, state=StateDescriptor(label="shift"))

        assert filter_fn(matching) is True
        assert filter_fn(non_matching) is False

    def test_get_bisegments_filter_by_transition_matches(self) -> None:
        baseline = StateDescriptor(label="baseline")
        StateDescriptor(label="shift")
        other = StateDescriptor(label="other")

        target_transition = _make_transition()
        filter_fn = NoResetClassificationTableByTransitionScenario._get_bisegments_filter_by_transition(
            target_transition
        )

        matching = BisegmentInfo(
            bisegment_num=0,
            bisegment_start=0,
            bisegment_end=5,
            change_point=3,
            transition=target_transition,
        )
        non_matching = BisegmentInfo(
            bisegment_num=0,
            bisegment_start=0,
            bisegment_end=5,
            change_point=3,
            transition=TransitionDescriptor(curr_state=baseline, next_state=other),
        )

        assert filter_fn(matching) is True
        assert filter_fn(non_matching) is False


# ---------------------------------------------------------------------------
# _get_arl_dataset
# ---------------------------------------------------------------------------


class TestGetArlDataset:
    def test_returns_state_dataset_on_success(self) -> None:
        scenario = NoResetClassificationTableByTransitionScenario(
            [_make_entry()],
            transition=_make_transition(),
            use_arl=True,
            arl_length=3,
        )
        dataset = Dataset([make_univariate_labeled(name="series")])
        result = scenario._get_arl_dataset(dataset, StateDescriptor(label="baseline"), 3)
        assert isinstance(result, StateDataset)
        assert result.state == StateDescriptor(label="baseline")
        assert len(result.timeseries) > 0

    def test_returns_empty_state_dataset_on_value_error(self, monkeypatch) -> None:
        scenario = NoResetClassificationTableByTransitionScenario(
            [_make_entry()],
            transition=_make_transition(),
            use_arl=True,
            arl_length=3,
        )

        def _fail(*args: object, **kwargs: object) -> StateDataset[object]:
            raise ValueError("no data")

        monkeypatch.setattr(StateDataset, "from_dataset", _fail)
        dataset = Dataset([make_univariate_labeled(name="series")])
        result = scenario._get_arl_dataset(dataset, StateDescriptor(label="baseline"), 3)
        assert isinstance(result, StateDataset)
        assert result.timeseries == []


# ---------------------------------------------------------------------------
# prepare_benchmark_jobs
# ---------------------------------------------------------------------------


class TestPrepareBenchmarkJobs:
    def test_without_arl_creates_one_job_per_entry(self) -> None:
        entry = _make_entry()
        scenario = NoResetClassificationTableByTransitionScenario(
            [entry],
            transition=_make_transition(),
            use_arl=False,
        )
        dataset = Dataset([make_univariate_labeled(name="series")])
        jobs = scenario.prepare_benchmark_jobs(dataset)
        assert len(jobs) == 1

    def test_without_arl_job_has_bisegment_providers(self) -> None:
        entry = _make_entry()
        scenario = NoResetClassificationTableByTransitionScenario(
            [entry],
            transition=_make_transition(),
            use_arl=False,
        )
        dataset = Dataset([make_univariate_labeled(name="series")])
        jobs = scenario.prepare_benchmark_jobs(dataset)
        assert len(jobs[0].providers) > 0

    def test_with_arl_creates_two_jobs(self) -> None:
        entry = _make_entry()
        scenario = NoResetClassificationTableByTransitionScenario(
            [entry],
            transition=_make_transition(),
            use_arl=True,
            arl_length=3,
        )
        dataset = Dataset([make_univariate_labeled(name="series")])
        jobs = scenario.prepare_benchmark_jobs(dataset)
        assert len(jobs) == 2

    def test_with_arl_and_empty_arl_skips_extra_job(self) -> None:
        entry = _make_entry()
        scenario = NoResetClassificationTableByTransitionScenario(
            [entry],
            transition=_make_transition(),
            use_arl=True,
            arl_length=50,
        )
        dataset = Dataset([make_univariate_labeled(name="series")])
        jobs = scenario.prepare_benchmark_jobs(dataset)
        assert len(jobs) == 1

    def test_with_arl_marks_has_arl_providers_true(self) -> None:
        entry = _make_entry()
        scenario = NoResetClassificationTableByTransitionScenario(
            [entry],
            transition=_make_transition(),
            use_arl=True,
            arl_length=3,
        )
        dataset = Dataset([make_univariate_labeled(name="series")])
        scenario.prepare_benchmark_jobs(dataset)
        assert scenario._has_arl_providers is True

    def test_with_arl_marks_has_arl_providers_false_when_empty(self) -> None:
        entry = _make_entry()
        scenario = NoResetClassificationTableByTransitionScenario(
            [entry],
            transition=_make_transition(),
            use_arl=True,
            arl_length=50,
        )
        dataset = Dataset([make_univariate_labeled(name="series")])
        scenario.prepare_benchmark_jobs(dataset)
        assert scenario._has_arl_providers is False

    def test_no_bisegment_match_returns_empty_job_providers(self) -> None:
        """When no bisegments match the transition, jobs have empty providers."""
        entry = _make_entry()
        scenario = NoResetClassificationTableByTransitionScenario(
            [entry],
            transition=_other_transition(),
            use_arl=False,
        )
        dataset = Dataset([make_univariate_labeled(name="series")])
        jobs = scenario.prepare_benchmark_jobs(dataset)
        assert len(jobs) == 1
        assert len(jobs[0].providers) == 0

    def test_multiple_entries_get_separate_jobs(self) -> None:
        entries = [_make_entry(), _make_entry()]
        scenario = NoResetClassificationTableByTransitionScenario(
            entries,
            transition=_make_transition(),
            use_arl=False,
        )
        dataset = Dataset([make_univariate_labeled(name="series")])
        jobs = scenario.prepare_benchmark_jobs(dataset)
        assert len(jobs) == 2


# ---------------------------------------------------------------------------
# set_classification_report / set_registry
# ---------------------------------------------------------------------------


class TestSetters:
    def test_set_classification_report_forwards_to_analyzer(self) -> None:
        scenario = NoResetClassificationTableByTransitionScenario(
            [_make_entry()],
            transition=_make_transition(),
            use_arl=False,
        )
        report = object()
        scenario.set_classification_report(report)
        assert scenario._classification_analyzer.classification_report is report

    def test_set_registry_updates_both_analyzers(self) -> None:
        scenario = NoResetClassificationTableByTransitionScenario(
            [_make_entry()],
            transition=_make_transition(),
            use_arl=True,
            arl_length=100,
        )
        registry: BenchmarkRegistry[float, object] = BenchmarkRegistry()
        scenario.set_registry(registry)
        assert scenario._classification_analyzer.registry is registry
        assert scenario._arl_analyzer.registry is registry


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_returns_dict_with_entry_descriptions(self, monkeypatch) -> None:
        entry = _make_entry()
        scenario = NoResetClassificationTableByTransitionScenario(
            [entry],
            transition=_make_transition(),
            use_arl=False,
        )
        scenario.set_classification_report(object())

        monkeypatch.setattr(
            scenario._classification_analyzer,
            "pick_runs",
            lambda entry, entries_picker=None: [],
        )
        monkeypatch.setattr(
            scenario._threshold_resolver,
            "resolve_classification_thresholds",
            lambda entry, runs, report: [0.5, 1.0],
        )
        monkeypatch.setattr(
            scenario._classification_analyzer,
            "analyze",
            lambda entry, thresholds, entries_picker=None: pd.DataFrame(
                {"threshold": [0.5, 1.0], "precision": [0.8, 0.9]}
            ),
        )

        registry: BenchmarkRegistry[float, object] = BenchmarkRegistry()
        result = scenario.analyze(registry)

        assert entry.description in result
        assert isinstance(result[entry.description], pd.DataFrame)

    def test_use_arl_appends_arl_column(self, monkeypatch) -> None:
        entry = _make_entry()
        scenario = NoResetClassificationTableByTransitionScenario(
            [entry],
            transition=_make_transition(),
            use_arl=True,
            arl_length=50,
        )
        scenario._has_arl_providers = True
        scenario.set_classification_report(object())

        monkeypatch.setattr(
            scenario._classification_analyzer,
            "pick_runs",
            lambda entry, entries_picker=None: [],
        )
        monkeypatch.setattr(
            scenario._threshold_resolver,
            "resolve_classification_thresholds",
            lambda entry, runs, report: [0.5, 1.0],
        )
        monkeypatch.setattr(
            scenario._classification_analyzer,
            "analyze",
            lambda entry, thresholds, entries_picker=None: pd.DataFrame(
                {"threshold": [0.5, 1.0], "precision": [0.8, 0.9]}
            ),
        )
        monkeypatch.setattr(
            scenario._arl_analyzer,
            "analyze",
            lambda entry, state, thresholds, arl_length: pd.DataFrame({"arl": [10.0, 20.0]}),
        )

        registry: BenchmarkRegistry[float, object] = BenchmarkRegistry()
        result = scenario.analyze(registry)

        df = result[entry.description]
        assert "arl" in df.columns
        assert list(df["arl"]) == [10.0, 20.0]

    def test_use_arl_with_no_arl_providers_skips_column(self, monkeypatch) -> None:
        entry = _make_entry()
        scenario = NoResetClassificationTableByTransitionScenario(
            [entry],
            transition=_make_transition(),
            use_arl=True,
            arl_length=50,
        )
        scenario._has_arl_providers = False
        scenario.set_classification_report(object())

        monkeypatch.setattr(
            scenario._classification_analyzer,
            "pick_runs",
            lambda entry, entries_picker=None: [],
        )
        monkeypatch.setattr(
            scenario._threshold_resolver,
            "resolve_classification_thresholds",
            lambda entry, runs, report: [0.5, 1.0],
        )
        monkeypatch.setattr(
            scenario._classification_analyzer,
            "analyze",
            lambda entry, thresholds, entries_picker=None: pd.DataFrame(
                {"threshold": [0.5, 1.0], "precision": [0.8, 0.9]}
            ),
        )

        registry: BenchmarkRegistry[float, object] = BenchmarkRegistry()
        result = scenario.analyze(registry)

        df = result[entry.description]
        assert "arl" not in df.columns

    def test_use_arl_with_empty_classification_table_skips_arl(self, monkeypatch) -> None:
        entry = _make_entry()
        scenario = NoResetClassificationTableByTransitionScenario(
            [entry],
            transition=_make_transition(),
            use_arl=True,
            arl_length=50,
        )
        scenario._has_arl_providers = True
        scenario.set_classification_report(object())

        monkeypatch.setattr(
            scenario._classification_analyzer,
            "pick_runs",
            lambda entry, entries_picker=None: [],
        )
        monkeypatch.setattr(
            scenario._threshold_resolver,
            "resolve_classification_thresholds",
            lambda entry, runs, report: [0.5, 1.0],
        )
        monkeypatch.setattr(
            scenario._classification_analyzer,
            "analyze",
            lambda entry, thresholds, entries_picker=None: pd.DataFrame(),
        )

        registry: BenchmarkRegistry[float, object] = BenchmarkRegistry()
        result = scenario.analyze(registry)

        df = result[entry.description]
        assert "arl" not in df.columns

    def test_multiple_entries_all_returned(self, monkeypatch) -> None:
        entry_a = _make_entry(threshold=3.0)
        entry_b = _make_entry(threshold=5.0)
        scenario = NoResetClassificationTableByTransitionScenario(
            [entry_a, entry_b],
            transition=_make_transition(),
            use_arl=False,
        )
        scenario.set_classification_report(object())

        monkeypatch.setattr(
            scenario._classification_analyzer,
            "pick_runs",
            lambda entry, entries_picker=None: [],
        )
        monkeypatch.setattr(
            scenario._threshold_resolver,
            "resolve_classification_thresholds",
            lambda entry, runs, report: [0.5],
        )
        monkeypatch.setattr(
            scenario._classification_analyzer,
            "analyze",
            lambda entry, thresholds, entries_picker=None: pd.DataFrame({"threshold": [0.5], "precision": [0.8]}),
        )

        registry: BenchmarkRegistry[float, object] = BenchmarkRegistry()
        result = scenario.analyze(registry)

        assert len(result) == 2
        assert entry_a.description in result
        assert entry_b.description in result
