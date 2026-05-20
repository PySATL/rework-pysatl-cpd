# -*- coding: ascii -*-
"""
Tests for benchmark outputs.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from pathlib import Path

from pysatl_cpd.analysis.metrics.abstracts.imultiple_run_metric import IMultipleRunMetric
from pysatl_cpd.benchmark.online.reset import OnlineResetBenchmark, OnlineResetBenchmarkEntry
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.core.online import OnlineDetectionTrace, OnlineResetDetector
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data import Dataset, LabeledData, TimeseriesAnnotation
from tests.support.algorithms import CountingAlgorithm, CountingAlgorithmConfig, CountingAlgorithmState
from tests.support.golden import load_json_golden
from tests.support.providers import make_univariate_labeled


class DetectedChangePointsMetric(
    IMultipleRunMetric[OnlineDetectionTrace[CountingAlgorithmState], LabeledData[float, TimeseriesAnnotation], int]
):
    def evaluate(
        self,
        runs: Sequence[
            SingleRun[OnlineDetectionTrace[CountingAlgorithmState], LabeledData[float, TimeseriesAnnotation]]
        ],
    ) -> int:
        return sum(len(run.trace.detected_change_points) for run in runs)


def test_online_reset_benchmark_matches_golden_result() -> None:
    golden = load_json_golden(
        Path(__file__).resolve().parents[1] / "golden" / "benchmarks" / "online_reset_totals.json"
    )
    dataset = Dataset(
        [
            make_univariate_labeled((1.0, 2.0, 3.0, 4.0), name="series-a"),
            make_univariate_labeled((2.0, 2.0, 2.0, 2.0), name="series-b"),
        ]
    )
    registry = BenchmarkRegistry[float, OnlineDetectionTrace[CountingAlgorithmState]]()
    benchmark = OnlineResetBenchmark(dataset, registry)
    detector = OnlineResetDetector(CountingAlgorithm(CountingAlgorithmConfig(threshold=5.0)), threshold=5.0)
    entry = OnlineResetBenchmarkEntry(detector)

    result = benchmark.get_metrics_for([entry], DetectedChangePointsMetric(), n_jobs=1)

    assert entry.description.name == golden["detector_name"]
    assert result[entry.description] == golden["result"]
