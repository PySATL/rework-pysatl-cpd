# -*- coding: ascii -*-
"""
Tests for pickers.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry
from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import ManualThresholdsRange
from pysatl_cpd.benchmark.online.noreset.tooling.bisegment_cut import BisegmentCut
from pysatl_cpd.benchmark.online.noreset.tooling.pickers import (
    OnlineNoResetBisegmentByTransitionPicker,
    OnlineNoResetEntryAlgorithmPicker,
    OnlineNoResetNoChangeByStatePicker,
)
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.single_run import SingleRunDescription
from pysatl_cpd.data.typedefs import (
    BisegmentAnnotation,
    NoChangeSeriesAnnotation,
    StateDescriptor,
    TransitionDescriptor,
)
from tests.support.algorithms import CountingAlgorithm, CountingAlgorithmConfig


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


def test_entry_algorithm_picker_keeps_only_matching_bisegments() -> None:
    picker = OnlineNoResetEntryAlgorithmPicker()
    benchmark_entry = _make_entry()
    expected = benchmark_entry.description

    matching_bisegment = SingleRunDescription(
        detector_description=expected,
        provider_description=BisegmentAnnotation(name="b1", transition=_make_transition()),
    )
    wrong_detector = SingleRunDescription(
        detector_description=ChangePointDetectorDescription(name="other"),
        provider_description=BisegmentAnnotation(name="b2", transition=_make_transition()),
    )
    wrong_provider_type = SingleRunDescription(
        detector_description=expected,
        provider_description=NoChangeSeriesAnnotation(name="nc", state=StateDescriptor(label="baseline")),
    )

    picked = picker.pick([matching_bisegment, wrong_detector, wrong_provider_type], benchmark_entry)

    assert picked == [matching_bisegment]


def test_bisegment_transition_picker_keeps_matching_transition_only() -> None:
    transition = _make_transition()
    other_transition = TransitionDescriptor(
        curr_state=StateDescriptor(label="foo"),
        next_state=StateDescriptor(label="bar"),
    )
    picker = OnlineNoResetBisegmentByTransitionPicker(transition)
    entry = _make_entry()
    expected = entry.description

    matching = SingleRunDescription(
        detector_description=expected,
        provider_description=BisegmentAnnotation(name="b1", transition=transition),
    )
    wrong_transition = SingleRunDescription(
        detector_description=expected,
        provider_description=BisegmentAnnotation(name="b2", transition=other_transition),
    )
    wrong_detector = SingleRunDescription(
        detector_description=ChangePointDetectorDescription(name="other"),
        provider_description=BisegmentAnnotation(name="b3", transition=transition),
    )

    picked = picker.pick([matching, wrong_transition, wrong_detector], entry)

    assert picked == [matching]


def test_nochange_state_picker_keeps_matching_state_only() -> None:
    state = StateDescriptor(label="baseline")
    other_state = StateDescriptor(label="shift")
    picker = OnlineNoResetNoChangeByStatePicker(state)
    entry = _make_entry()
    expected = entry.description

    matching = SingleRunDescription(
        detector_description=expected,
        provider_description=NoChangeSeriesAnnotation(name="n1", state=state),
    )
    wrong_state = SingleRunDescription(
        detector_description=expected,
        provider_description=NoChangeSeriesAnnotation(name="n2", state=other_state),
    )
    wrong_detector = SingleRunDescription(
        detector_description=ChangePointDetectorDescription(name="other"),
        provider_description=NoChangeSeriesAnnotation(name="n3", state=state),
    )

    picked = picker.pick([matching, wrong_state, wrong_detector], entry)

    assert picked == [matching]


def test_entry_algorithm_picker_matches_entry_description() -> None:
    picker = OnlineNoResetEntryAlgorithmPicker()
    entry = _make_entry()
    expected = entry.description

    matching = SingleRunDescription(
        detector_description=expected,
        provider_description=BisegmentAnnotation(name="b1", transition=_make_transition()),
    )
    wrong = SingleRunDescription(
        detector_description=ChangePointDetectorDescription(name="other"),
        provider_description=BisegmentAnnotation(name="b2", transition=_make_transition()),
    )

    picked = picker.pick([matching, wrong], entry)

    assert picked == [matching]


def test_entry_description_depends_on_bisegment_cut() -> None:
    entry_no_cut = _make_entry()
    entry_cut = OnlineNoResetBenchmarkEntry(
        algorithm=CountingAlgorithm(CountingAlgorithmConfig(threshold=3.0)),
        thresholds=ManualThresholdsRange(_values=(1.0, 2.0)),
        bisegment_cut=BisegmentCut(left_trim=2, right_trim=1),
    )

    assert entry_no_cut.description != entry_cut.description
