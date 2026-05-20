"""
Tests for state arl comprehensive.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import logging

import pytest

from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry
from pysatl_cpd.benchmark.online.noreset.scenarios.state_arl import NoResetArlByStateScenario
from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import ManualThresholdsRange
from pysatl_cpd.data.typedefs import SegmentInfo, StateDescriptor
from tests.support.algorithms import CountingAlgorithm, CountingAlgorithmConfig
from tests.support.providers import make_univariate_labeled


def _make_entry() -> OnlineNoResetBenchmarkEntry:
    return OnlineNoResetBenchmarkEntry(
        algorithm=CountingAlgorithm(CountingAlgorithmConfig()),
        thresholds=ManualThresholdsRange(_values=(1.0, 2.0)),
    )


# ---------------------------------------------------------------------------
# __init__ - additional validation not covered by existing tests
# ---------------------------------------------------------------------------


class TestInitAdditional:
    def test_rejects_zero_arl_length(self) -> None:
        with pytest.raises(ValueError, match="arl_length must be positive"):
            NoResetArlByStateScenario(
                [_make_entry()],
                state=StateDescriptor(label="baseline"),
                arl_length=0,
            )

    def test_rejects_negative_arl_length(self) -> None:
        with pytest.raises(ValueError, match="arl_length must be positive"):
            NoResetArlByStateScenario(
                [_make_entry()],
                state=StateDescriptor(label="baseline"),
                arl_length=-1,
            )

    def test_accepts_positive_arl_length(self) -> None:
        scenario = NoResetArlByStateScenario(
            [_make_entry()],
            state=StateDescriptor(label="baseline"),
            arl_length=10,
        )
        assert scenario.arl_length == 10

    def test_has_arl_providers_initially_false(self) -> None:
        scenario = NoResetArlByStateScenario(
            [_make_entry()],
            state=StateDescriptor(label="baseline"),
            arl_length=10,
        )
        assert scenario._has_arl_providers is False


# ---------------------------------------------------------------------------
# state_checked property
# ---------------------------------------------------------------------------


class TestStateChecked:
    def test_returns_state_when_set(self) -> None:
        state = StateDescriptor(label="baseline")
        scenario = NoResetArlByStateScenario(
            [_make_entry()],
            state=state,
            arl_length=10,
        )
        assert scenario.state_checked == state

    def test_raises_when_state_is_none(self) -> None:
        scenario = NoResetArlByStateScenario(
            [_make_entry()],
            state=StateDescriptor(label="baseline"),
            arl_length=10,
        )
        scenario.state = None
        with pytest.raises(ValueError, match="state is required"):
            _ = scenario.state_checked


# ---------------------------------------------------------------------------
# prepare_benchmark_jobs - warning path (empty ARL dataset)
# ---------------------------------------------------------------------------


class TestPrepareBenchmarkJobsWarning:
    def test_logs_warning_when_arl_dataset_empty(self, caplog) -> None:
        caplog.set_level(logging.WARNING)
        scenario = NoResetArlByStateScenario(
            [_make_entry()],
            state=StateDescriptor(label="baseline"),
            arl_length=50,
        )
        dataset = type(
            "Dataset",
            (),
            {
                "timeseries": [make_univariate_labeled(name="series")],
                "filter_by_segments": lambda self, filter_fn: type(
                    "Filtered",
                    (),
                    {
                        "timeseries": [make_univariate_labeled(name="series")],
                        "merge": lambda self: make_univariate_labeled(name="series"),
                    },
                )(),
                "merge": lambda self: make_univariate_labeled(name="series"),
            },
        )()

        jobs = scenario.prepare_benchmark_jobs(dataset)
        assert len(jobs) == 1
        assert scenario._has_arl_providers is False

    def test_warning_message_contains_arl_length(self, caplog) -> None:
        caplog.set_level(logging.WARNING)
        scenario = NoResetArlByStateScenario(
            [_make_entry()],
            state=StateDescriptor(label="baseline"),
            arl_length=50,
        )
        dataset = type(
            "Dataset",
            (),
            {
                "timeseries": [make_univariate_labeled(name="series")],
                "filter_by_segments": lambda self, filter_fn: type(
                    "Filtered",
                    (),
                    {
                        "timeseries": [make_univariate_labeled(name="series")],
                        "merge": lambda self: make_univariate_labeled(name="series"),
                    },
                )(),
                "merge": lambda self: make_univariate_labeled(name="series"),
            },
        )()

        scenario.prepare_benchmark_jobs(dataset)

        assert any("arl_length=50" in record.message for record in caplog.records)
        assert any("no ARL runs will be registered" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# _get_segments_filter_by_state
# ---------------------------------------------------------------------------


class TestSegmentsFilter:
    def test_matches_correctly(self) -> None:
        filter_fn = NoResetArlByStateScenario._get_segments_filter_by_state(StateDescriptor(label="baseline"))
        matching = SegmentInfo(segment_num=0, segment_start=0, segment_end=2, state=StateDescriptor(label="baseline"))
        non_matching = SegmentInfo(segment_num=1, segment_start=3, segment_end=5, state=StateDescriptor(label="shift"))
        assert filter_fn(matching) is True
        assert filter_fn(non_matching) is False

    def test_equality_on_state_object(self) -> None:
        state_a = StateDescriptor(label="baseline")
        state_b = StateDescriptor(label="baseline")
        filter_fn = NoResetArlByStateScenario._get_segments_filter_by_state(state_a)
        segment = SegmentInfo(segment_num=0, segment_start=0, segment_end=2, state=state_b)
        assert filter_fn(segment) is True
