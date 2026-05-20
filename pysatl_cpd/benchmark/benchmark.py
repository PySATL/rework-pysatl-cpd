# -*- coding: ascii -*-
"""Benchmark orchestrator for PySATL CPD."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pathlib import Path

from pysatl_cpd.benchmark.registry import DEFAULT_JOB_PARALLEL_BACKEND, BenchmarkRegistry
from pysatl_cpd.benchmark.scenarios import BenchmarkScenario
from pysatl_cpd.core import DetectionTrace
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.data import TimeseriesAnnotation
from pysatl_cpd.data.dataset import Dataset


class Benchmark[DataT, TraceT: DetectionTrace]:
    """Orchestrates benchmark execution over a dataset through a detector registry.

    Manages the full lifecycle of running multiple detector configurations
    against a dataset, persisting intermediate results in an associated
    registry, and collecting per-scenario analysis at the end.

    Parameters
    ----------
    dataset
        Labeled dataset whose providers serve as detector inputs.
    registry
        Registry that caches and retrieves per-detector execution results.
    n_jobs
        Number of parallel worker processes (default 1). Must be non-zero.
    """

    def __init__(
        self,
        dataset: Dataset[DataT, TimeseriesAnnotation],
        registry: BenchmarkRegistry[DataT, TraceT],
        *,
        n_jobs: int = 1,
    ) -> None:
        self._dataset = dataset
        self._registry = registry
        self.n_jobs = n_jobs

    @property
    def dataset(self) -> Dataset[DataT, TimeseriesAnnotation]:
        """Dataset bound to this benchmark instance."""
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: Dataset[DataT, TimeseriesAnnotation]) -> None:
        """Replace the dataset used for benchmarking."""
        self._dataset = dataset

    @property
    def registry(self) -> BenchmarkRegistry[DataT, TraceT]:
        """Registry of precomputed detector executions bound to this benchmark."""
        return self._registry

    @property
    def n_jobs(self) -> int:
        """Number of parallel worker processes used for benchmark execution."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs: int) -> None:
        """Set the number of parallel worker processes.

        Parameters
        ----------
        n_jobs
            Worker count. Must be non-zero.

        Raises
        ------
        ValueError
            If n_jobs is zero.
        """
        if n_jobs == 0:
            raise ValueError("n_jobs must be non-zero")
        self._n_jobs = n_jobs

    def upload_registry(self, upload_registry_path: Path) -> None:
        """Load a previously exported registry file from disk.

        Parameters
        ----------
        upload_registry_path
            Filesystem path to a pickled registry file.
        """
        self._registry.upload_registry(upload_registry_path)

    def export_registry(self, export_registry_path: Path) -> None:
        """Persist the current registry contents to disk as a pickle.

        Parameters
        ----------
        export_registry_path
            Destination filesystem path for the pickled registry.
        """
        self._registry.export_registry(export_registry_path)

    def run_scenario[ResultT](
        self,
        scenario: BenchmarkScenario[DataT, TraceT, ResultT],
        *,
        force_recompute: bool = False,
        n_jobs: int | None = None,
        backend: str = DEFAULT_JOB_PARALLEL_BACKEND,
    ) -> dict[ChangePointDetectorDescription, ResultT]:
        """Execute a benchmark scenario across all detectors in the registry.

        Iterates over jobs prepared by the scenario, runs each detector
        against its assigned providers, and returns the aggregated scenario
        analysis. Entries already present in the registry are skipped unless
        ``force_recompute`` is set.

        Parameters
        ----------
        scenario
            Scenario defining job preparation and result analysis logic.
        force_recompute
            If True, re-executes detectors even when a cached result exists.
        n_jobs
            Worker count override; falls back to instance-level n_jobs when
            None.
        backend
            Joblib parallel backend identifier (default ``"loky"``; e.g. ``threading``).

        Returns
        -------
        dict[pysatl_cpd.core.change_point_detector.ChangePointDetectorDescription, ResultT]
            Scenario analysis results keyed by detector description.
        """
        resolved_n_jobs = self._n_jobs if n_jobs is None else n_jobs
        for job in scenario.prepare_benchmark_jobs(self._dataset):
            if not job.providers:
                continue
            try:
                self._registry.update(
                    job.detector,
                    job.providers,
                    force_recompute=force_recompute,
                    n_jobs=resolved_n_jobs,
                    backend=backend,
                )
            except ValueError as exc:
                scenario.handle_benchmark_error(job, exc)

        return scenario.analyze(self._registry)
