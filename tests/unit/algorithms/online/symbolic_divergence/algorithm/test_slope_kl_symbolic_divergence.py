# -*- coding: ascii -*-
"""
Tests for slope kl symbolic divergence.
"""

from __future__ import annotations

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math

import pytest

from pysatl_cpd.algorithms.online import SlopeKLSymbolicDivergence
from pysatl_cpd.algorithms.online.symbolic_divergence.slope_kl_symbolic_divergence import (
    SlopeKLSymbolicDivergenceConfiguration,
    SlopeKLSymbolicDivergenceState,
)
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.raw import RawDivergenceStatistic
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.scaled import ScaledDivergenceStatistic


class TestSlopeKLSymbolicDivergenceConfiguration:
    def test_rejects_non_positive_learning_period_size(self) -> None:
        with pytest.raises(ValueError, match="learning_period_size"):
            SlopeKLSymbolicDivergenceConfiguration(learning_period_size=0, delta=0.0, gamma=1.0)

    def test_identical_configs_have_same_hash(self) -> None:
        config_a = SlopeKLSymbolicDivergenceConfiguration(learning_period_size=10, delta=0.1, gamma=1.0, smoothing=1e-9)
        config_b = SlopeKLSymbolicDivergenceConfiguration(learning_period_size=10, delta=0.1, gamma=1.0, smoothing=1e-9)
        assert hash(config_a) == hash(config_b)
        assert config_a == config_b

    @pytest.mark.parametrize(
        "overrides",
        [
            {"delta": 0.2},
            {"gamma": 2.0},
            {"smoothing": 1e-6},
            {"learning_period_size": 20},
        ],
    )
    def test_different_component_params_have_different_hash(self, overrides: dict) -> None:
        base = {"learning_period_size": 10, "delta": 0.1, "gamma": 1.0, "smoothing": 1e-9}
        config_a = SlopeKLSymbolicDivergenceConfiguration(**base)
        config_b = SlopeKLSymbolicDivergenceConfiguration(**{**base, **overrides})
        assert hash(config_a) != hash(config_b)


class TestSlopeKLSymbolicDivergenceConstruction:
    def test_rejects_learning_period_not_greater_than_window(self) -> None:
        with pytest.raises(ValueError, match="must be strictly"):
            SlopeKLSymbolicDivergence(learning_period_size=2, delta=0.0, gamma=1.0)

    def test_delegates_encoder_validation(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            SlopeKLSymbolicDivergence(learning_period_size=5, delta=1.0, gamma=1.0)

    def test_defaults_to_scaled_two_n_d_statistic(self) -> None:
        algorithm = SlopeKLSymbolicDivergence(learning_period_size=5, delta=0.0, gamma=1.0)
        assert isinstance(algorithm.statistic, ScaledDivergenceStatistic)

    def test_explicit_statistic_is_respected(self) -> None:
        statistic = RawDivergenceStatistic()
        algorithm = SlopeKLSymbolicDivergence(learning_period_size=5, delta=0.0, gamma=1.0, statistic=statistic)
        assert algorithm.statistic is statistic


class TestSlopeKLSymbolicDivergence:
    def test_name_and_types(self) -> None:
        algorithm = SlopeKLSymbolicDivergence(learning_period_size=5, delta=0.0, gamma=1.0)
        assert algorithm.name == "SlopeKLSymbolicDivergence"
        assert isinstance(algorithm.configuration, SlopeKLSymbolicDivergenceConfiguration)
        assert isinstance(algorithm.state, SlopeKLSymbolicDivergenceState)

    def test_learning_period_emits_zero_then_finite_scores(self) -> None:
        algorithm = SlopeKLSymbolicDivergence(learning_period_size=4, delta=0.0, gamma=1.0)
        emitted = [algorithm.process(value) for value in [0.0, 1.0, 2.0, 3.0, 10.0, -10.0]]
        assert emitted[:4] == [0.0, 0.0, 0.0, 0.0]
        assert all(math.isfinite(float(value)) for value in emitted)

    def test_default_value_equals_two_n_times_raw_divergence(self) -> None:
        observations = [0.0, 1.0, 2.0, 3.0, 4.0, 10.0, -5.0, 8.0]
        scaled = SlopeKLSymbolicDivergence(learning_period_size=4, delta=0.0, gamma=1.0)
        raw = SlopeKLSymbolicDivergence(
            learning_period_size=4, delta=0.0, gamma=1.0, statistic=RawDivergenceStatistic()
        )

        scaled_value = 0.0
        raw_value = 0.0
        for observation in observations:
            scaled_value = float(scaled.process(observation))
            raw_value = float(raw.process(observation))

        sample_size = sum(scaled.symbol_counts)
        assert scaled_value == pytest.approx(2.0 * sample_size * raw_value)
