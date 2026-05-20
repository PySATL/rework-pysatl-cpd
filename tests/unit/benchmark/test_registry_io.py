# -*- coding: ascii -*-
"""
Tests for registry io.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pickle
from pathlib import Path
from typing import Any

import pytest

from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.single_run import SingleRun, SingleRunDescription
from pysatl_cpd.data.typedefs import TimeseriesAnnotation
from tests.support.algorithms import RecordingDetector
from tests.support.providers import make_univariate_labeled


def _populated_registry() -> BenchmarkRegistry[Any, Any]:
    registry: BenchmarkRegistry[Any, Any] = BenchmarkRegistry()
    detector = RecordingDetector()
    registry.update(
        detector,
        [make_univariate_labeled(data=[1.0, 2.0], name="series-a")],
        n_jobs=1,
    )
    return registry


def test_executions_registry_returns_internal_dict() -> None:
    registry = _populated_registry()
    mapping = registry.executions_registry
    assert len(mapping) == 1
    key = next(iter(mapping))
    assert key in mapping
    assert isinstance(mapping[key], SingleRun)


def test_contains_present() -> None:
    registry = _populated_registry()
    key = next(iter(registry.executions_registry))
    assert key in registry


def test_contains_missing() -> None:
    registry = _populated_registry()
    missing = SingleRunDescription(
        detector_description=ChangePointDetectorDescription(name="other"),
        provider_description=TimeseriesAnnotation(name="other"),
    )
    assert missing not in registry


def test_len() -> None:
    registry = _populated_registry()
    assert len(registry) == 1
    registry.update(
        RecordingDetector(),
        [make_univariate_labeled(data=[3.0, 4.0], name="series-b")],
        n_jobs=1,
    )
    assert len(registry) == 2


def test_keys() -> None:
    registry = _populated_registry()
    keys = registry.keys()
    key = next(iter(registry.executions_registry))
    assert list(keys) == [key]


def test_values() -> None:
    registry = _populated_registry()
    values = registry.values()
    run = next(iter(values))
    assert isinstance(run, SingleRun)


def test_items() -> None:
    registry = _populated_registry()
    items = list(registry.items())
    assert len(items) == 1
    key, value = items[0]
    assert isinstance(key, SingleRunDescription)
    assert isinstance(value, SingleRun)


def test_getitem() -> None:
    registry = _populated_registry()
    key = next(iter(registry.executions_registry))
    run = registry[key]
    assert isinstance(run, SingleRun)
    assert run is registry._executions_registry[key]


def test_update_skips_existing_key() -> None:
    registry = _populated_registry()
    assert len(registry) == 1
    original_key = next(iter(registry.executions_registry))
    original_run = registry[original_key]
    detector = RecordingDetector()
    registry.update(
        detector,
        [make_univariate_labeled(data=[1.0, 2.0], name="series-a")],
        n_jobs=1,
    )
    assert len(registry) == 1
    assert registry[original_key] is original_run


def test_export_registry_writes_pickle(tmp_path: Path) -> None:
    registry = _populated_registry()
    export_path = tmp_path / "registry.pkl"
    registry.export_registry(export_path)
    assert export_path.exists()
    with export_path.open("rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, dict)
    assert len(loaded) == 1


def test_export_registry_creates_parent_dirs(tmp_path: Path) -> None:
    registry = _populated_registry()
    export_path = tmp_path / "subdir" / "nested" / "registry.pkl"
    registry.export_registry(export_path)
    assert export_path.exists()
    with export_path.open("rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, dict)


def test_upload_registry_loads_valid(tmp_path: Path) -> None:
    original = _populated_registry()
    path = tmp_path / "export.pkl"
    original.export_registry(path)

    fresh: BenchmarkRegistry[Any, Any] = BenchmarkRegistry()
    fresh.upload_registry(path)

    assert len(fresh) == 1
    assert list(fresh.keys()) == list(original.keys())


def test_upload_registry_raises_type_error_for_non_dict(tmp_path: Path) -> None:
    path = tmp_path / "not_a_dict.pkl"
    with path.open("wb") as f:
        pickle.dump([1, 2, 3], f)

    registry: BenchmarkRegistry[Any, Any] = BenchmarkRegistry()
    with pytest.raises(TypeError, match="Registry file must contain a dict"):
        registry.upload_registry(path)


def test_upload_registry_raises_type_error_for_wrong_entries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong_types.pkl"
    bad_dict: dict[str, str] = {"not_a_description": "not_a_run"}
    with path.open("wb") as f:
        pickle.dump(bad_dict, f)

    registry: BenchmarkRegistry[Any, Any] = BenchmarkRegistry()
    with pytest.raises(TypeError, match="Registry file entries must be SingleRunDescription"):
        registry.upload_registry(path)
