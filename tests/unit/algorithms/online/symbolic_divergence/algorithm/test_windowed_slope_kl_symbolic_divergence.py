# -*- coding: ascii -*-
"""
Tests for windowed slope kl symbolic divergence.
"""

from __future__ import annotations

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math

import pytest

from pysatl_cpd.algorithms.online import WindowedSlopeKLSymbolicDivergence
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.log_scaled import LogScaledDivergenceStatistic
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.raw import RawDivergenceStatistic
from pysatl_cpd.algorithms.online.symbolic_divergence.windowed_slope_kl_symbolic_divergence import (
    WindowedSlopeKLSymbolicDivergenceConfiguration,
    WindowedSlopeKLSymbolicDivergenceState,
)


class TestWindowedSlopeKLSymbolicDivergenceConfiguration:
    def test_rejects_non_positive_learning_period_size(self) -> None:
        with pytest.raises(ValueError, match="learning_period_size"):
            WindowedSlopeKLSymbolicDivergenceConfiguration(
                learning_period_size=0, recent_window_size=4, delta=0.0, gamma=1.0
            )

    def test_rejects_non_positive_recent_window_size(self) -> None:
        with pytest.raises(ValueError, match="recent_window_size"):
            WindowedSlopeKLSymbolicDivergenceConfiguration(
                learning_period_size=4, recent_window_size=0, delta=0.0, gamma=1.0
            )

    def test_identical_configs_have_same_hash(self) -> None:
        config_a = WindowedSlopeKLSymbolicDivergenceConfiguration(
            learning_period_size=10, recent_window_size=8, delta=0.1, gamma=1.0, smoothing=1e-9
        )
        config_b = WindowedSlopeKLSymbolicDivergenceConfiguration(
            learning_period_size=10, recent_window_size=8, delta=0.1, gamma=1.0, smoothing=1e-9
        )
        assert hash(config_a) == hash(config_b)
        assert config_a == config_b

    @pytest.mark.parametrize(
        "overrides",
        [
            {"delta": 0.2},
            {"gamma": 2.0},
            {"smoothing": 1e-6},
            {"learning_period_size": 20},
            {"recent_window_size": 12},
        ],
    )
    def test_different_component_params_have_different_hash(self, overrides: dict) -> None:
        base = {"learning_period_size": 10, "recent_window_size": 8, "delta": 0.1, "gamma": 1.0, "smoothing": 1e-9}
        config_a = WindowedSlopeKLSymbolicDivergenceConfiguration(**base)
        config_b = WindowedSlopeKLSymbolicDivergenceConfiguration(**{**base, **overrides})
        assert hash(config_a) != hash(config_b)


class TestWindowedSlopeKLSymbolicDivergenceConstruction:
    def test_rejects_learning_period_not_greater_than_window(self) -> None:
        with pytest.raises(ValueError, match="must be strictly"):
            WindowedSlopeKLSymbolicDivergence(learning_period_size=2, recent_window_size=4, delta=0.0, gamma=1.0)

    def test_rejects_recent_window_not_greater_than_window(self) -> None:
        with pytest.raises(ValueError, match="must be strictly"):
            WindowedSlopeKLSymbolicDivergence(learning_period_size=4, recent_window_size=2, delta=0.0, gamma=1.0)

    def test_delegates_encoder_validation(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            WindowedSlopeKLSymbolicDivergence(learning_period_size=5, recent_window_size=5, delta=1.0, gamma=1.0)

    def test_defaults_to_raw_statistic(self) -> None:
        algorithm = WindowedSlopeKLSymbolicDivergence(
            learning_period_size=5, recent_window_size=5, delta=0.0, gamma=1.0
        )
        assert isinstance(algorithm.statistic, RawDivergenceStatistic)

    def test_explicit_statistic_is_respected(self) -> None:
        statistic = LogScaledDivergenceStatistic()
        algorithm = WindowedSlopeKLSymbolicDivergence(
            learning_period_size=5, recent_window_size=5, delta=0.0, gamma=1.0, statistic=statistic
        )
        assert algorithm.statistic is statistic


class TestWindowedSlopeKLSymbolicDivergence:
    def test_name_and_types(self) -> None:
        algorithm = WindowedSlopeKLSymbolicDivergence(
            learning_period_size=5, recent_window_size=5, delta=0.0, gamma=1.0
        )
        assert algorithm.name == "WindowedSlopeKLSymbolicDivergence"
        assert isinstance(algorithm.configuration, WindowedSlopeKLSymbolicDivergenceConfiguration)
        assert isinstance(algorithm.state, WindowedSlopeKLSymbolicDivergenceState)

    def test_learning_period_emits_zero_then_finite_scores(self) -> None:
        algorithm = WindowedSlopeKLSymbolicDivergence(
            learning_period_size=4, recent_window_size=4, delta=0.0, gamma=1.0
        )
        emitted = [algorithm.process(value) for value in [0.0, 1.0, 2.0, 3.0, 10.0, -10.0, 5.0, -2.0, 7.0, 1.0]]
        assert emitted[:4] == [0.0, 0.0, 0.0, 0.0]
        assert all(math.isfinite(float(value)) for value in emitted)

    def test_recent_window_counts_are_bounded(self) -> None:
        algorithm = WindowedSlopeKLSymbolicDivergence(
            learning_period_size=4, recent_window_size=4, delta=0.0, gamma=1.0
        )
        for value in [0.0, 1.0, 2.0, 3.0, 10.0, -10.0, 5.0, -2.0, 7.0, 1.0, 3.0, 9.0]:
            algorithm.process(value)
        maxlen = 4 - 2 + 1
        assert sum(algorithm.symbol_counts) == maxlen

    def test_state_reference_distribution_after_learning(self) -> None:
        algorithm = WindowedSlopeKLSymbolicDivergence(
            learning_period_size=4, recent_window_size=4, delta=0.0, gamma=1.0
        )
        for value in [0.0, 1.0, 2.0, 3.0]:
            algorithm.process(value)
        state = algorithm.state
        assert state.reference_distribution != ()
        assert len(state.reference_distribution) == algorithm.encoder.num_symbols
