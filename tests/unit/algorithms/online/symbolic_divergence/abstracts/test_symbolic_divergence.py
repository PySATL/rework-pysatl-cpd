# -*- coding: ascii -*-
"""
Tests for symbolic divergence.
"""

from __future__ import annotations

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.algorithms.online.symbolic_divergence.divergences.kl_divergence import KLDivergence
from pysatl_cpd.algorithms.online.symbolic_divergence.encoders.slope_encoder import SlopeEncoder
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.raw import RawDivergenceStatistic
from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.scaled import ScaledDivergenceStatistic
from pysatl_cpd.algorithms.online.symbolic_divergence.symbolic_divergence import (
    SymbolicDivergence,
    SymbolicDivergenceConfiguration,
    SymbolicDivergenceState,
)


class _MinimalSymbolicDivergence(SymbolicDivergence[SymbolicDivergenceConfiguration, SymbolicDivergenceState]):
    def __init__(self, learning_period_size: int = 3, statistic=None) -> None:
        super().__init__(
            configuration=SymbolicDivergenceConfiguration(learning_period_size=learning_period_size),
            encoder=SlopeEncoder(delta=0.0, gamma=1.0),
            divergence=KLDivergence(),
            statistic=statistic,
        )

    @property
    def name(self) -> str:
        return "_MinimalSymbolicDivergence"

    @property
    def configuration(self) -> SymbolicDivergenceConfiguration:
        return self._configuration

    @property
    def state(self) -> SymbolicDivergenceState:
        return SymbolicDivergenceState(
            is_in_learning_period=self.is_in_learning_period,
            samples_count=self.samples_count,
            symbol_counts=self.symbol_counts,
            reference_distribution=self.reference_distribution,
        )


class TestSymbolicDivergenceConfiguration:
    def test_rejects_non_positive_learning_period_size(self) -> None:
        with pytest.raises(ValueError, match="learning_period_size"):
            SymbolicDivergenceConfiguration(learning_period_size=0)

    def test_identical_configs_have_same_hash(self) -> None:
        config_a = SymbolicDivergenceConfiguration(learning_period_size=10)
        config_b = SymbolicDivergenceConfiguration(learning_period_size=10)
        assert hash(config_a) == hash(config_b)
        assert config_a == config_b

    def test_different_configs_have_different_hash(self) -> None:
        config_a = SymbolicDivergenceConfiguration(learning_period_size=5)
        config_b = SymbolicDivergenceConfiguration(learning_period_size=10)
        assert hash(config_a) != hash(config_b)


class TestSymbolicDivergenceState:
    def test_empty_state_is_hashable(self) -> None:
        state = SymbolicDivergenceState(is_in_learning_period=True)
        assert isinstance(hash(state), int)

    def test_identical_states_have_same_hash(self) -> None:
        state_a = SymbolicDivergenceState(is_in_learning_period=False, samples_count=3, symbol_counts=(1, 2))
        state_b = SymbolicDivergenceState(is_in_learning_period=False, samples_count=3, symbol_counts=(1, 2))
        assert hash(state_a) == hash(state_b)


class TestSymbolicDivergenceConstruction:
    def test_cannot_instantiate_abstract_base(self) -> None:
        with pytest.raises(TypeError):
            SymbolicDivergence(  # type: ignore[abstract]
                configuration=SymbolicDivergenceConfiguration(learning_period_size=3),
                encoder=SlopeEncoder(delta=0.0, gamma=1.0),
                divergence=KLDivergence(),
            )

    def test_rejects_learning_period_not_greater_than_window(self) -> None:
        with pytest.raises(ValueError, match="must be strictly"):
            _MinimalSymbolicDivergence(learning_period_size=2)

    def test_defaults_to_raw_statistic(self) -> None:
        algorithm = _MinimalSymbolicDivergence()
        assert isinstance(algorithm.statistic, RawDivergenceStatistic)

    def test_injected_components_are_exposed(self) -> None:
        algorithm = _MinimalSymbolicDivergence(statistic=ScaledDivergenceStatistic())
        assert isinstance(algorithm.encoder, SlopeEncoder)
        assert isinstance(algorithm.divergence, KLDivergence)
        assert isinstance(algorithm.statistic, ScaledDivergenceStatistic)


class TestSymbolicDivergenceProcess:
    def test_learning_period_emits_zero_then_scores(self) -> None:
        algorithm = _MinimalSymbolicDivergence(learning_period_size=3)
        emitted = [algorithm.process(value) for value in [0.0, 2.0, 4.0, 6.0, 0.0]]
        assert emitted[0] == 0.0
        assert emitted[1] == 0.0
        assert emitted[2] == 0.0
        assert all(isinstance(value, float) for value in emitted)

    def test_reference_distribution_set_after_learning(self) -> None:
        algorithm = _MinimalSymbolicDivergence(learning_period_size=3)
        for value in [0.0, 2.0, 4.0]:
            algorithm.process(value)
        assert algorithm.reference_distribution != ()
        assert len(algorithm.reference_distribution) == algorithm.encoder.num_symbols

    def test_symbol_counts_accumulate(self) -> None:
        algorithm = _MinimalSymbolicDivergence(learning_period_size=3)
        for value in [0.0, 2.0, 4.0]:
            algorithm.process(value)
        assert sum(algorithm.symbol_counts) == 2


class TestSymbolicDivergenceReset:
    def test_reset_restores_fresh_state(self) -> None:
        algorithm = _MinimalSymbolicDivergence(learning_period_size=3)
        for value in [0.0, 2.0, 4.0, 6.0]:
            algorithm.process(value)
        algorithm.reset()
        assert algorithm.samples_count == 0
        assert algorithm.is_in_learning_period is True
        assert algorithm.reference_distribution == ()
        assert sum(algorithm.symbol_counts) == 0


class TestSymbolicDivergenceRecreate:
    def test_recreate_returns_independent_fresh_instance(self) -> None:
        algorithm = _MinimalSymbolicDivergence(learning_period_size=3)
        for value in [0.0, 2.0, 4.0, 6.0]:
            algorithm.process(value)

        recreated = algorithm.recreate()

        assert recreated is not algorithm
        assert recreated.samples_count == 0
        assert recreated.is_in_learning_period is True
