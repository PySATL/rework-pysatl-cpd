# -*- coding: ascii -*-
"""
Tests for scenarios base.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, cast

import pytest

from pysatl_cpd.benchmark.scenarios import BenchmarkJob, BenchmarkScenario


class _ConcreteScenario(BenchmarkScenario[Any, Any, Any]):
    """Minimal concrete scenario to test base-class behaviour."""

    def prepare_benchmark_jobs(self, dataset: Any) -> Any:
        return []

    def analyze(self, registry: Any) -> Any:
        return {}


def test_handle_benchmark_error_re_raises() -> None:
    scenario = _ConcreteScenario()
    job = cast(BenchmarkJob[Any], None)
    with pytest.raises(ValueError, match="test error"):
        scenario.handle_benchmark_error(job, ValueError("test error"))
