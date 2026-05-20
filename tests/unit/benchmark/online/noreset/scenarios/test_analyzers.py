"""
Tests for analyzers.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.benchmark.online.noreset.scenarios.classification_table_global import (
    NoResetClassificationTableScenario,
)
from pysatl_cpd.benchmark.online.noreset.scenarios.classification_table_transition import (
    NoResetClassificationTableByTransitionScenario,
)
from pysatl_cpd.benchmark.online.noreset.scenarios.individual_bisegments_table import NoResetBisegmentsTableScenario
from pysatl_cpd.benchmark.online.noreset.scenarios.state_arl import NoResetArlByStateScenario
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.core.online import OnlineDetectionTrace
from pysatl_cpd.data.typedefs import StateDescriptor, TransitionDescriptor

# ---------------------------------------------------------------------------
# NoResetClassificationTableScenario
# ---------------------------------------------------------------------------


class TestClassificationTableScenarioAnalyzers:
    """Only creates a ClassificationTableAnalyzer - no others."""

    def test_creates_classification_analyzer(self) -> None:
        scenario = NoResetClassificationTableScenario([])
        assert hasattr(scenario, "_classification_analyzer")
        # It should NOT have extraneous analyzers
        assert not hasattr(scenario, "_arl_analyzer")
        assert not hasattr(scenario, "_bisegment_analyzer")


# ---------------------------------------------------------------------------
# NoResetClassificationTableByTransitionScenario
# ---------------------------------------------------------------------------


class TestTransitionScenarioAnalyzers:
    """Creates both classification and ARL analyzers sharing injected state."""

    @pytest.fixture
    def transition(self) -> TransitionDescriptor:
        state = StateDescriptor(type="test")
        return TransitionDescriptor(curr_state=state, next_state=state)

    def test_creates_both_analyzers(self, transition: TransitionDescriptor) -> None:
        scenario = NoResetClassificationTableByTransitionScenario(
            [],
            collect_states=False,
            transition=transition,
            use_arl=True,
            arl_length=100,
        )
        assert hasattr(scenario, "_classification_analyzer")
        assert hasattr(scenario, "_arl_analyzer")

    def test_no_extraneous_analyzers(self, transition: TransitionDescriptor) -> None:
        scenario = NoResetClassificationTableByTransitionScenario(
            [],
            collect_states=False,
            transition=transition,
            use_arl=True,
            arl_length=100,
        )
        assert not hasattr(scenario, "_bisegment_analyzer")

    def test_analyzers_share_registry(self, transition: TransitionDescriptor) -> None:
        scenario = NoResetClassificationTableByTransitionScenario(
            [],
            collect_states=False,
            transition=transition,
            use_arl=True,
            arl_length=100,
        )
        registry: BenchmarkRegistry[float, OnlineDetectionTrace[object]] = BenchmarkRegistry()
        scenario.set_registry(registry)
        assert scenario._classification_analyzer.registry is registry
        assert scenario._arl_analyzer.registry is registry

    def test_analyzer_raises_before_set_registry(self, transition: TransitionDescriptor) -> None:
        scenario = NoResetClassificationTableByTransitionScenario(
            [],
            collect_states=False,
            transition=transition,
            use_arl=True,
            arl_length=100,
        )
        with pytest.raises(ValueError, match="Registry is not set"):
            _ = scenario._classification_analyzer.registry

    def test_use_arl_false_does_not_require_arl_length(self) -> None:
        state = StateDescriptor(type="test")
        transition = TransitionDescriptor(curr_state=state, next_state=state)
        scenario = NoResetClassificationTableByTransitionScenario(
            [],
            collect_states=False,
            transition=transition,
            use_arl=False,
        )
        assert scenario.use_arl is False


# ---------------------------------------------------------------------------
# NoResetBisegmentsTableScenario
# ---------------------------------------------------------------------------


class TestBisegmentsTableScenarioAnalyzers:
    """Only creates a BisegmentAnalyzer - no others."""

    def test_creates_bisegment_analyzer(self) -> None:
        scenario = NoResetBisegmentsTableScenario([], threshold=0.5)
        assert hasattr(scenario, "_bisegment_analyzer")
        assert not hasattr(scenario, "_classification_analyzer")
        assert not hasattr(scenario, "_arl_analyzer")

    def test_handle_benchmark_error_preserved(self) -> None:
        """Verify handle_benchmark_error still works (didn't change during refactoring)."""
        scenario = NoResetBisegmentsTableScenario([], threshold=0.5)
        assert hasattr(scenario, "handle_benchmark_error")


# ---------------------------------------------------------------------------
# NoResetArlByStateScenario
# ---------------------------------------------------------------------------


class TestArlByStateScenarioAnalyzers:
    """Only creates an ArlAnalyzer - no others."""

    def test_creates_arl_analyzer(self) -> None:
        scenario = NoResetArlByStateScenario(
            [],
            state=StateDescriptor(type="test"),
            arl_length=100,
        )
        assert hasattr(scenario, "_arl_analyzer")
        assert not hasattr(scenario, "_classification_analyzer")
        assert not hasattr(scenario, "_bisegment_analyzer")

    def test_raises_value_error_without_state(self) -> None:
        with pytest.raises(ValueError, match="state is required"):
            NoResetArlByStateScenario([], state=None, arl_length=100)
