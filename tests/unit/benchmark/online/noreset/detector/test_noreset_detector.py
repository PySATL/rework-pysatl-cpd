# -*- coding: ascii -*-
"""
Tests for noreset detector.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np

from pysatl_cpd.benchmark.online.noreset.detector.noreset_detector import NoResetOnlineDetector
from pysatl_cpd.benchmark.online.noreset.tooling.bisegment_cut import BisegmentCut
from tests.support.algorithms import CountingAlgorithm, CountingAlgorithmConfig, IdentityTransformer
from tests.support.providers import make_univariate_provider


class TestNoResetOnlineDetectorClone:
    def test_clone_creates_independent_instance(self) -> None:
        algo = CountingAlgorithm(CountingAlgorithmConfig())
        detector = NoResetOnlineDetector(algo, collect_states=True)
        cloned = detector.clone()

        assert cloned is not detector
        assert isinstance(cloned, NoResetOnlineDetector)

    def test_clone_recreates_algorithm(self) -> None:
        algo = CountingAlgorithm(CountingAlgorithmConfig())
        detector = NoResetOnlineDetector(algo, collect_states=True)
        cloned = detector.clone()

        assert cloned.algorithm is not detector.algorithm
        assert cloned.algorithm.configuration == detector.algorithm.configuration

    def test_clone_preserves_collect_states(self) -> None:
        algo = CountingAlgorithm(CountingAlgorithmConfig())
        detector = NoResetOnlineDetector(algo, collect_states=False)
        cloned = detector.clone()

        assert cloned._collect_states == detector._collect_states

    def test_clone_with_data_transformer(self) -> None:
        transformer = IdentityTransformer()
        algo = CountingAlgorithm(CountingAlgorithmConfig())
        detector = NoResetOnlineDetector(algo, collect_states=True, data_transformer=transformer)
        cloned = detector.clone()

        assert cloned.data_transformer is not None
        assert cloned.data_transformer.annotation == transformer.annotation

    def test_clone_without_data_transformer(self) -> None:
        algo = CountingAlgorithm(CountingAlgorithmConfig())
        detector = NoResetOnlineDetector(algo, collect_states=True, data_transformer=None)
        cloned = detector.clone()

        assert cloned.data_transformer is None

    def test_clone_independent_algorithm_state(self) -> None:
        algo = CountingAlgorithm(CountingAlgorithmConfig())
        detector = NoResetOnlineDetector(algo, collect_states=True)
        cloned = detector.clone()

        algo.reset()
        cloned.algorithm.reset()

        algo.process(10.0)
        cloned.algorithm.process(20.0)

        assert algo.state.running_total == 10.0
        assert cloned.algorithm.state.running_total == 20.0

    def test_clone_returns_detector_type(self) -> None:
        algo = CountingAlgorithm(CountingAlgorithmConfig())
        detector = NoResetOnlineDetector(algo, collect_states=True)
        cloned = detector.clone()

        assert cloned.description.name == "no_reset_online_detector"

    def test_clone_preserves_bisegment_cut(self) -> None:
        algo = CountingAlgorithm(CountingAlgorithmConfig())
        detector = NoResetOnlineDetector(
            algo,
            collect_states=True,
            bisegment_cut=BisegmentCut(left_trim=1, right_trim=2),
        )
        cloned = detector.clone()

        assert cloned.description.parameters["left_trim"] == 1
        assert cloned.description.parameters["right_trim"] == 2


class TestNoResetOnlineDetectorBisegmentCut:
    def test_description_contains_zero_trim_by_default(self) -> None:
        detector = NoResetOnlineDetector(CountingAlgorithm(CountingAlgorithmConfig()))
        assert detector.description.parameters["left_trim"] == 0
        assert detector.description.parameters["right_trim"] == 0

    def test_detect_with_noop_bisegment_cut(self) -> None:
        detector = NoResetOnlineDetector(
            CountingAlgorithm(CountingAlgorithmConfig()),
            collect_states=True,
        )
        provider = make_univariate_provider(data=(1.0, 2.0, 3.0), name="noop-series")

        trace = detector.detect(provider)

        assert len(trace.detection_function) == len(provider)
        assert trace.detection_function.tolist() == [1.0, 3.0, 6.0]
        assert trace.skip_periods == []
        assert all(s is not None for s in trace.algorithm_states)

    def test_detect_with_bisegment_cut_preserves_length_and_global_indices(self) -> None:
        detector = NoResetOnlineDetector(
            CountingAlgorithm(CountingAlgorithmConfig()),
            collect_states=True,
            bisegment_cut=BisegmentCut(left_trim=1, right_trim=2),
        )
        provider = make_univariate_provider(data=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), name="trimmed-series")

        trace = detector.detect(provider)

        assert len(trace.detection_function) == len(provider)
        assert trace.skip_periods == [(0, 0), (4, 5)]
        assert np.isnan(trace.detection_function[0])
        assert trace.detection_function[1:4].tolist() == [2.0, 5.0, 9.0]
        assert np.isnan(trace.detection_function[4])
        assert np.isnan(trace.detection_function[5])
        assert trace.algorithm_states[0] is None
        assert trace.algorithm_states[4] is None
        assert trace.algorithm_states[5] is None
