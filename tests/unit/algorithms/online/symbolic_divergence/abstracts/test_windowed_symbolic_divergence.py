# -*- coding: ascii -*-
"""
Tests for windowed symbolic divergence.
"""

from __future__ import annotations

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math

import pytest

from pysatl_cpd.algorithms.online.symbolic_divergence.divergences.kl_divergence import KLDivergence
from pysatl_cpd.algorithms.online.symbolic_divergence.encoders.slope_encoder import SlopeEncoder
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.log_scaled import LogScaledDivergenceStatistic
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.raw import RawDivergenceStatistic
from pysatl_cpd.algorithms.online.symbolic_divergence.windowed_symbolic_divergence import (
    WindowedSymbolicDivergence,
    WindowedSymbolicDivergenceConfiguration,
    WindowedSymbolicDivergenceState,
)


class _MinimalWindowedSymbolicDivergence(
    WindowedSymbolicDivergence[WindowedSymbolicDivergenceConfiguration, WindowedSymbolicDivergenceState]
):
    def __init__(self, learning_period_size: int = 3, recent_window_size: int = 3, statistic=None) -> None:
        super().__init__(
            configuration=WindowedSymbolicDivergenceConfiguration(
                learning_period_size=learning_period_size,
                recent_window_size=recent_window_size,
            ),
            encoder=SlopeEncoder(delta=0.0, gamma=1.0),
            divergence=KLDivergence(),
            statistic=statistic,
        )

    @property
    def name(self) -> str:
        return "_MinimalWindowedSymbolicDivergence"

    @property
    def configuration(self) -> WindowedSymbolicDivergenceConfiguration:
        return self._configuration

    @property
    def state(self) -> WindowedSymbolicDivergenceState:
        return WindowedSymbolicDivergenceState(
            is_in_learning_period=self.is_in_learning_period,
            samples_count=self.samples_count,
            symbol_counts=self.symbol_counts,
            reference_distribution=self.reference_distribution,
        )


class TestWindowedSymbolicDivergenceConfiguration:
    def test_rejects_non_positive_learning_period_size(self) -> None:
        with pytest.raises(ValueError, match="learning_period_size"):
            WindowedSymbolicDivergenceConfiguration(learning_period_size=0, recent_window_size=3)

    def test_rejects_non_positive_recent_window_size(self) -> None:
        with pytest.raises(ValueError, match="recent_window_size"):
            WindowedSymbolicDivergenceConfiguration(learning_period_size=3, recent_window_size=0)

    def test_identical_configs_have_same_hash(self) -> None:
        config_a = WindowedSymbolicDivergenceConfiguration(learning_period_size=10, recent_window_size=8)
        config_b = WindowedSymbolicDivergenceConfiguration(learning_period_size=10, recent_window_size=8)
        assert hash(config_a) == hash(config_b)
        assert config_a == config_b

    @pytest.mark.parametrize("overrides", [{"learning_period_size": 5}, {"recent_window_size": 4}])
    def test_different_configs_have_different_hash(self, overrides: dict) -> None:
        base = {"learning_period_size": 10, "recent_window_size": 8}
        config_a = WindowedSymbolicDivergenceConfiguration(**base)
        config_b = WindowedSymbolicDivergenceConfiguration(**{**base, **overrides})
        assert hash(config_a) != hash(config_b)


class TestWindowedSymbolicDivergenceState:
    def test_empty_state_is_hashable(self) -> None:
        state = WindowedSymbolicDivergenceState(is_in_learning_period=True)
        assert isinstance(hash(state), int)

    def test_identical_states_have_same_hash(self) -> None:
        state_a = WindowedSymbolicDivergenceState(is_in_learning_period=False, samples_count=3, symbol_counts=(1, 2))
        state_b = WindowedSymbolicDivergenceState(is_in_learning_period=False, samples_count=3, symbol_counts=(1, 2))
        assert hash(state_a) == hash(state_b)


class TestWindowedSymbolicDivergenceConstruction:
    def test_cannot_instantiate_abstract_base(self) -> None:
        with pytest.raises(TypeError):
            WindowedSymbolicDivergence(  # type: ignore[abstract]
                configuration=WindowedSymbolicDivergenceConfiguration(learning_period_size=3, recent_window_size=3),
                encoder=SlopeEncoder(delta=0.0, gamma=1.0),
                divergence=KLDivergence(),
            )

    def test_rejects_learning_period_not_greater_than_window(self) -> None:
        with pytest.raises(ValueError, match="must be strictly"):
            _MinimalWindowedSymbolicDivergence(learning_period_size=2, recent_window_size=4)

    def test_rejects_recent_window_not_greater_than_window(self) -> None:
        with pytest.raises(ValueError, match="must be strictly"):
            _MinimalWindowedSymbolicDivergence(learning_period_size=4, recent_window_size=2)

    def test_defaults_to_raw_statistic(self) -> None:
        algorithm = _MinimalWindowedSymbolicDivergence()
        assert isinstance(algorithm.statistic, RawDivergenceStatistic)

    def test_injected_components_are_exposed(self) -> None:
        algorithm = _MinimalWindowedSymbolicDivergence(statistic=LogScaledDivergenceStatistic())
        assert isinstance(algorithm.encoder, SlopeEncoder)
        assert isinstance(algorithm.divergence, KLDivergence)
        assert isinstance(algorithm.statistic, LogScaledDivergenceStatistic)


class TestWindowedSymbolicDivergenceProcess:
    def test_learning_period_emits_zero(self) -> None:
        algorithm = _MinimalWindowedSymbolicDivergence(learning_period_size=3, recent_window_size=3)
        emitted = [algorithm.process(value) for value in [0.0, 2.0, 4.0]]
        assert emitted == [0.0, 0.0, 0.0]
        assert all(isinstance(value, float) for value in emitted)

    def test_reference_distribution_set_after_learning(self) -> None:
        algorithm = _MinimalWindowedSymbolicDivergence(learning_period_size=3, recent_window_size=3)
        for value in [0.0, 2.0, 4.0]:
            algorithm.process(value)
        assert algorithm.reference_distribution != ()
        assert len(algorithm.reference_distribution) == algorithm.encoder.num_symbols

    def test_symbol_counts_reset_when_reference_is_fixed(self) -> None:
        algorithm = _MinimalWindowedSymbolicDivergence(learning_period_size=3, recent_window_size=3)
        for value in [0.0, 2.0, 4.0]:
            algorithm.process(value)
        assert sum(algorithm.symbol_counts) == 0

    def test_emits_finite_scores_after_recent_window_fills(self) -> None:
        algorithm = _MinimalWindowedSymbolicDivergence(learning_period_size=3, recent_window_size=3)
        emitted = [algorithm.process(value) for value in [0.0, 2.0, 4.0, 6.0, 0.0, 5.0, -5.0, 2.0]]
        assert any(value != 0.0 for value in emitted)
        assert all(math.isfinite(float(value)) for value in emitted)

    def test_emits_zero_until_recent_window_fills(self) -> None:
        algorithm = _MinimalWindowedSymbolicDivergence(learning_period_size=3, recent_window_size=4)
        emitted = [algorithm.process(value) for value in [0.0, 2.0, 4.0, 5.0, 4.0, 8.0, 7.0]]
        assert emitted[:6] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert emitted[6] != 0.0

    def test_reference_distribution_updates_when_window_slides(self) -> None:
        algorithm = _MinimalWindowedSymbolicDivergence(learning_period_size=3, recent_window_size=4)
        for value in [0.0, 2.0, 4.0, 5.0, 4.0, 8.0, 7.0]:
            algorithm.process(value)

        reference_before = algorithm.reference_distribution
        assert reference_before != ()

        algorithm.process(9.0)
        assert algorithm.reference_distribution != reference_before

    def test_recent_window_counts_are_bounded(self) -> None:
        algorithm = _MinimalWindowedSymbolicDivergence(learning_period_size=3, recent_window_size=3)
        for value in [0.0, 2.0, 4.0, 6.0, 0.0, 5.0, -5.0, 2.0, 7.0, -1.0]:
            algorithm.process(value)
        maxlen = 3 - 2 + 1
        assert sum(algorithm.symbol_counts) == maxlen


class TestWindowedSymbolicDivergenceReset:
    def test_reset_restores_fresh_state(self) -> None:
        algorithm = _MinimalWindowedSymbolicDivergence(learning_period_size=3, recent_window_size=3)
        for value in [0.0, 2.0, 4.0, 6.0, 0.0, 5.0]:
            algorithm.process(value)
        algorithm.reset()
        assert algorithm.samples_count == 0
        assert algorithm.is_in_learning_period is True
        assert algorithm.reference_distribution == ()
        assert sum(algorithm.symbol_counts) == 0

    def test_reset_resets_statistic_component(self) -> None:
        statistic = LogScaledDivergenceStatistic()
        statistic.update(1.0, 1)
        algorithm = _MinimalWindowedSymbolicDivergence(
            learning_period_size=3, recent_window_size=4, statistic=statistic
        )
        assert statistic.value != 0.0

        algorithm.reset()
        assert statistic.value == 0.0


class TestWindowedSymbolicDivergenceRecreate:
    def test_recreate_returns_independent_fresh_instance(self) -> None:
        algorithm = _MinimalWindowedSymbolicDivergence(learning_period_size=3, recent_window_size=3)
        for value in [0.0, 2.0, 4.0, 6.0, 0.0, 5.0]:
            algorithm.process(value)

        recreated = algorithm.recreate()

        assert recreated is not algorithm
        assert recreated.samples_count == 0
        assert recreated.is_in_learning_period is True
