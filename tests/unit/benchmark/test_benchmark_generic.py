# -*- coding: ascii -*-
"""
Tests for benchmark generic.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pathlib import Path
from typing import Any, cast

import pytest

from pysatl_cpd.benchmark.benchmark import Benchmark
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.benchmark.scenarios import BenchmarkScenario
from pysatl_cpd.data import TimeseriesAnnotation
from pysatl_cpd.data.dataset import Dataset


def test_dataset_property() -> None:
    dataset = Dataset[Any, TimeseriesAnnotation](cast(Any, []))
    registry: BenchmarkRegistry[Any, Any] = BenchmarkRegistry()
    benchmark = Benchmark(dataset, registry)
    assert benchmark.dataset is dataset


def test_dataset_setter() -> None:
    dataset_a = Dataset[Any, TimeseriesAnnotation](cast(Any, []))
    dataset_b = Dataset[Any, TimeseriesAnnotation](cast(Any, []))
    registry: BenchmarkRegistry[Any, Any] = BenchmarkRegistry()
    benchmark = Benchmark(dataset_a, registry)
    benchmark.dataset = dataset_b
    assert benchmark.dataset is dataset_b


def test_registry_property() -> None:
    dataset = Dataset[Any, TimeseriesAnnotation](cast(Any, []))
    registry: BenchmarkRegistry[Any, Any] = BenchmarkRegistry()
    benchmark = Benchmark(dataset, registry)
    assert benchmark.registry is registry


def test_n_jobs_default() -> None:
    dataset = Dataset[Any, TimeseriesAnnotation](cast(Any, []))
    registry: BenchmarkRegistry[Any, Any] = BenchmarkRegistry()
    benchmark = Benchmark(dataset, registry)
    assert benchmark.n_jobs == 1


def test_n_jobs_getter() -> None:
    dataset = Dataset[Any, TimeseriesAnnotation](cast(Any, []))
    registry: BenchmarkRegistry[Any, Any] = BenchmarkRegistry()
    benchmark = Benchmark(dataset, registry, n_jobs=4)
    assert benchmark.n_jobs == 4


def test_n_jobs_setter() -> None:
    dataset = Dataset[Any, TimeseriesAnnotation](cast(Any, []))
    registry: BenchmarkRegistry[Any, Any] = BenchmarkRegistry()
    benchmark = Benchmark(dataset, registry, n_jobs=1)
    benchmark.n_jobs = 8
    assert benchmark.n_jobs == 8


def test_n_jobs_rejects_zero() -> None:
    dataset = Dataset[Any, TimeseriesAnnotation](cast(Any, []))
    registry: BenchmarkRegistry[Any, Any] = BenchmarkRegistry()
    benchmark = Benchmark(dataset, registry, n_jobs=1)
    with pytest.raises(ValueError, match="n_jobs must be non-zero"):
        benchmark.n_jobs = 0


def test_upload_registry_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = Dataset[Any, TimeseriesAnnotation](cast(Any, []))
    registry: BenchmarkRegistry[Any, Any] = BenchmarkRegistry()
    benchmark = Benchmark(dataset, registry)
    path = Path("/some/path")
    calls: list[Path] = []

    monkeypatch.setattr(registry, "upload_registry", lambda p: calls.append(p))

    benchmark.upload_registry(path)

    assert calls == [path]


def test_export_registry_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = Dataset[Any, TimeseriesAnnotation](cast(Any, []))
    registry: BenchmarkRegistry[Any, Any] = BenchmarkRegistry()
    benchmark = Benchmark(dataset, registry)
    path = Path("/some/path")
    calls: list[Path] = []

    monkeypatch.setattr(registry, "export_registry", lambda p: calls.append(p))

    benchmark.export_registry(path)

    assert calls == [path]


def test_run_scenario_handles_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = Dataset[Any, TimeseriesAnnotation](cast(Any, []))
    registry: BenchmarkRegistry[Any, Any] = BenchmarkRegistry()
    benchmark = Benchmark(dataset, registry)

    class _MockJob:
        providers = [cast(Any, None)]
        detector = cast(Any, None)

    mock_job = _MockJob()
    handled: list[tuple[Any, Exception]] = []

    class _ErrorHandlingScenario(BenchmarkScenario[Any, Any, dict[str, bool]]):
        def prepare_benchmark_jobs(self, dataset: Any) -> Any:
            return [mock_job]

        def analyze(self, registry: Any) -> dict[str, bool]:
            return {"done": True}

        def handle_benchmark_error(self, job: Any, exc: ValueError) -> None:
            handled.append((job, exc))

    def failing_update(*args: Any, **kwargs: Any) -> None:
        raise ValueError("registry error")

    monkeypatch.setattr(registry, "update", failing_update)

    result = benchmark.run_scenario(_ErrorHandlingScenario())

    assert result == {"done": True}
    assert len(handled) == 1
    assert handled[0][0] is mock_job
    assert isinstance(handled[0][1], ValueError)
    assert str(handled[0][1]) == "registry error"


def test_run_scenario_skips_jobs_with_empty_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset[Any, TimeseriesAnnotation](cast(Any, []))
    registry: BenchmarkRegistry[Any, Any] = BenchmarkRegistry()
    benchmark = Benchmark(dataset, registry)

    class _MockJob:
        providers: list[Any] = []
        detector = cast(Any, None)

    class _SkipScenario(BenchmarkScenario[Any, Any, dict[str, bool]]):
        def prepare_benchmark_jobs(self, dataset: Any) -> Any:
            return [_MockJob()]

        def analyze(self, registry: Any) -> dict[str, bool]:
            return {"skipped": True}

    update_called = False

    def spy_update(*args: Any, **kwargs: Any) -> None:
        nonlocal update_called
        update_called = True

    monkeypatch.setattr(registry, "update", spy_update)

    result = benchmark.run_scenario(_SkipScenario())

    assert result == {"skipped": True}
    assert not update_called
