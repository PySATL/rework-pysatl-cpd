# -*- coding: ascii -*-
"""Scenarios for reset-online benchmark execution and analysis."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from pysatl_cpd.analysis.metrics.abstracts.imultiple_run_metric import IMultipleRunMetric
from pysatl_cpd.benchmark.online.reset.entry import OnlineResetBenchmarkEntry
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.benchmark.scenarios import BenchmarkJob, BenchmarkScenario
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.online import OnlineDetectionTrace
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data import LabeledData, TimeseriesAnnotation
from pysatl_cpd.data.dataset import Dataset


@dataclass
class OnlineResetWholeTimeseriesMetricScenario[DataT, ResultT](
    BenchmarkScenario[DataT, OnlineDetectionTrace[Any], ResultT],
):
    """A scenario that evaluates a multiple-run metric across whole timeseries.

    For each detector entry, runs detection on every provider in the
    dataset and then evaluates the supplied metric over the collected
    set of runs.
    """

    entries: Sequence[OnlineResetBenchmarkEntry]
    metric: IMultipleRunMetric[OnlineDetectionTrace[Any], LabeledData[DataT, TimeseriesAnnotation], ResultT]
    _provider_annotations: set[TimeseriesAnnotation] = field(default_factory=set, init=False, repr=False)

    def prepare_benchmark_jobs(
        self,
        dataset: Dataset[DataT, TimeseriesAnnotation],
    ) -> Sequence[BenchmarkJob[DataT]]:
        """Create one job per entry, each running against all dataset providers.

        Also records the set of provider annotations for later filtering
        in ``analyze``.

        Parameters
        ----------
        dataset
            Dataset whose providers are used as detector inputs.

        Returns
        -------
        Sequence[BenchmarkJob]
            One job per entry, each with the full provider list.
        """
        providers = list(dataset.timeseries)
        self._provider_annotations = {provider.annotation for provider in providers}
        return [BenchmarkJob(entry.detector.clone(), providers) for entry in self.entries]

    def analyze(
        self,
        registry: BenchmarkRegistry[DataT, OnlineDetectionTrace[Any]],
    ) -> dict[ChangePointDetectorDescription, ResultT]:
        """Evaluate the metric for each detector entry using registry data.

        Filters registry values to match the entry's detector description
        and the provider annotations recorded during preparation, then
        evaluates the metric over the matching runs.

        Parameters
        ----------
        registry
            Registry containing cached execution results.

        Returns
        -------
        dict[ChangePointDetectorDescription, ResultT]
            Metric evaluation results keyed by detector description.
        """
        result: dict[ChangePointDetectorDescription, ResultT] = {}
        for entry in self.entries:
            runs = [
                run
                for run in registry.values()
                if run.trace.detector_description == entry.description
                and run.provider.annotation in self._provider_annotations
            ]
            result[entry.description] = self.metric.evaluate(
                cast(Sequence[SingleRun[OnlineDetectionTrace[Any], LabeledData[DataT, TimeseriesAnnotation]]], runs),
            )
        return result
