# -*- coding: ascii -*-
"""Benchmark scenario definitions."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.core import ChangePointDetector, DetectionTrace
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.data import LabeledData, TimeseriesAnnotation
from pysatl_cpd.data.dataset import Dataset


@dataclass(frozen=True)
class BenchmarkJob[DataT]:
    """A frozen dataclass binding a detector to its assigned providers.

    Attributes
    ----------
    detector : ChangePointDetector
        Detector configuration to benchmark.
    providers : Sequence[LabeledData]
        Input data providers the detector will be executed against.
    """

    detector: ChangePointDetector[DataT]
    providers: Sequence[LabeledData[DataT, TimeseriesAnnotation]]


class BenchmarkScenario[DataT, TraceT: DetectionTrace, ResultT](ABC):
    """Abstract definition of a benchmark scenario.

    A scenario encapsulates the lifecycle of a benchmark campaign:
    preparing detector-provider jobs from a dataset, orchestrating
    their execution through a registry, and producing analysis results
    from the accumulated registry data.
    """

    @abstractmethod
    def prepare_benchmark_jobs(  # pragma: no cover
        self,
        dataset: Dataset[DataT, TimeseriesAnnotation],
    ) -> Sequence[BenchmarkJob[DataT]]:
        """Produce the sequence of detector-provider jobs for this scenario.

        Parameters
        ----------
        dataset
            Dataset whose providers and detector configurations are used
            to build the benchmark jobs.

        Returns
        -------
        Sequence[BenchmarkJob]
            Jobs to be registered and executed against the registry.
        """

    @abstractmethod
    def analyze(
        self, registry: BenchmarkRegistry[DataT, TraceT]
    ) -> dict[ChangePointDetectorDescription, ResultT]:  # pragma: no cover
        """Derive scenario results from the populated registry.

        Parameters
        ----------
        registry
            Registry containing execution results for all jobs previously
            prepared by this scenario.

        Returns
        -------
        dict[ChangePointDetectorDescription, ResultT]
            Scenario-specific analysis results keyed by detector description.
        """

    def handle_benchmark_error(self, job: BenchmarkJob[DataT], exc: ValueError) -> None:
        """Handle a ValueError raised during benchmark job execution.

        Default implementation re-raises the exception. Override in
        subclasses to implement custom error recovery or logging.

        Parameters
        ----------
        job
            The job that raised the error.
        exc
            The exception instance.

        Raises
        ------
        ValueError
            Always re-raises the supplied exception.
        """
        raise ValueError(str(exc)) from None
