# -*- coding: ascii -*-
"""Reset-online benchmark orchestrator for whole-timeseries metrics."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.metrics.abstracts.imultiple_run_metric import IMultipleRunMetric
from pysatl_cpd.benchmark.benchmark import Benchmark
from pysatl_cpd.benchmark.online.reset.entry import OnlineResetBenchmarkEntry
from pysatl_cpd.benchmark.online.reset.scenarios import OnlineResetWholeTimeseriesMetricScenario
from pysatl_cpd.benchmark.registry import DEFAULT_JOB_PARALLEL_BACKEND, BenchmarkRegistry
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.online import OnlineDetectionTrace
from pysatl_cpd.data import LabeledData, TimeseriesAnnotation
from pysatl_cpd.data.dataset import Dataset


class OnlineResetBenchmark[DataT](Benchmark[DataT, OnlineDetectionTrace[Any]]):
    """Benchmark subclass specialised for reset-online detectors.

    Wraps the generic ``Benchmark`` and provides a convenience method
    for evaluating multiple detectors against a whole-timeseries metric.

    Parameters
    ----------
    dataset
        Labeled dataset whose providers serve as detector inputs.
    registry
        Registry that caches per-detector execution results.
    n_jobs
        Number of parallel worker processes (default 1). Must be non-zero.
    """

    def __init__(
        self,
        dataset: Dataset[DataT, TimeseriesAnnotation],
        registry: BenchmarkRegistry[DataT, OnlineDetectionTrace[Any]],
        *,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(dataset, registry, n_jobs=n_jobs)

    def get_metrics_for[ResultT](
        self,
        entries: Sequence[OnlineResetBenchmarkEntry],
        metric: IMultipleRunMetric[OnlineDetectionTrace[Any], LabeledData[DataT, TimeseriesAnnotation], ResultT],
        *,
        force_recompute: bool = False,
        n_jobs: int | None = None,
        backend: str = DEFAULT_JOB_PARALLEL_BACKEND,
    ) -> dict[ChangePointDetectorDescription, ResultT]:
        """Run a whole-timeseries metric for a collection of detector entries.

        Creates an ``OnlineResetWholeTimeseriesMetricScenario`` from the
        supplied entries and metric, then delegates to ``run_scenario``.

        Parameters
        ----------
        entries
            Detector entries to benchmark.
        metric
            Metric that evaluates multiple detection runs collectively.
        force_recompute
            If True, re-executes detectors even when cached results exist.
        n_jobs
            Worker count override; falls back to instance n_jobs when None.
        backend
            Joblib parallel backend identifier (default ``"loky"``).

        Returns
        -------
        dict[pysatl_cpd.core.change_point_detector.ChangePointDetectorDescription, ResultT]
            Metric results keyed by detector description.
        """
        return self.run_scenario(
            OnlineResetWholeTimeseriesMetricScenario(entries, metric),
            force_recompute=force_recompute,
            n_jobs=n_jobs,
            backend=backend,
        )
