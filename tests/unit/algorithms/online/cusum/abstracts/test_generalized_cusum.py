# -*- coding: ascii -*-
"""
Tests for generalized cusum.
"""

from __future__ import annotations

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.algorithms.online.cusum.abstracts.changepoint_func import ICusumChangepointFunc
from pysatl_cpd.algorithms.online.cusum.abstracts.estimator import IEstimatingSchema, ISchemaEstimates
from pysatl_cpd.algorithms.online.cusum.abstracts.generalized_cusum import (
    GeneralizedCUSUM,
    GeneralizedCUSUMConfiguration,
    GeneralizedCUSUMState,
)
from pysatl_cpd.algorithms.online.cusum.abstracts.monitoring import IMonitoringSchema


class _Estimates(ISchemaEstimates, total=False):
    mean: np.ndarray


class _Estimator(IEstimatingSchema[np.ndarray, _Estimates]):
    def __init__(self) -> None:
        self._estimates: _Estimates = {}

    def train(self, train_set) -> None:
        self._estimates = {"mean": np.mean(np.asarray(train_set, dtype=np.float64), axis=0)}

    def update(self, observation: np.ndarray) -> None:
        self._estimates = {"mean": np.asarray(observation, dtype=np.float64)}

    def reset(self) -> None:
        self._estimates = {}

    @property
    def estimates(self) -> _Estimates:
        return self._estimates


class _Monitoring(IMonitoringSchema[np.ndarray, _Estimates, np.ndarray]):
    def evaluate(self, observation: np.ndarray, estimates: _Estimates) -> np.ndarray:
        return np.asarray(observation, dtype=np.float64) - estimates["mean"]

    def reset(self) -> None:
        pass


class _ChangepointFunc(ICusumChangepointFunc[np.ndarray]):
    def __init__(self) -> None:
        self._value = 0.0

    def update(self, observation: np.ndarray) -> None:
        self._value += float(np.sum(observation))

    @property
    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._value = 0.0


class _MinimalCUSUM(
    GeneralizedCUSUM[np.ndarray, GeneralizedCUSUMConfiguration, GeneralizedCUSUMState, _Estimates, np.ndarray]
):
    def __init__(self, learning_period_size: int = 2, adaptive_estimation: bool = True) -> None:
        super().__init__(
            configuration=GeneralizedCUSUMConfiguration(
                learning_period_size=learning_period_size,
                adaptive_estimation=adaptive_estimation,
            ),
            estimating_schema=_Estimator(),
            monitoring_schema=_Monitoring(),
            changepoint_func=_ChangepointFunc(),
        )

    @property
    def name(self) -> str:
        return "_MinimalCUSUM"

    @property
    def configuration(self) -> GeneralizedCUSUMConfiguration:
        return self._config

    @property
    def state(self) -> GeneralizedCUSUMState:
        return GeneralizedCUSUMState(is_in_learning_period=self._is_training, statistics=self.estimates)


class TestGeneralizedCUSUMConfiguration:
    def test_identical_configs_have_same_hash(self) -> None:
        config_a = GeneralizedCUSUMConfiguration(
            learning_period_size=10,
            adaptive_estimation=True,
        )
        config_b = GeneralizedCUSUMConfiguration(
            learning_period_size=10,
            adaptive_estimation=True,
        )
        assert hash(config_a) == hash(config_b)
        assert config_a == config_b

    def test_different_configs_have_different_hash(self) -> None:
        config_a = GeneralizedCUSUMConfiguration(
            learning_period_size=5,
            adaptive_estimation=True,
        )
        config_b = GeneralizedCUSUMConfiguration(
            learning_period_size=10,
            adaptive_estimation=True,
        )
        assert hash(config_a) != hash(config_b)


class TestGeneralizedCUSUMState:
    def test_state_with_empty_statistics_is_hashable(self) -> None:
        state = GeneralizedCUSUMState(
            is_in_learning_period=True,
            statistics={},
        )
        h = hash(state)
        assert isinstance(h, int)

    def test_identical_states_have_same_hash(self) -> None:
        state_a = GeneralizedCUSUMState(
            is_in_learning_period=False,
            statistics={},
        )
        state_b = GeneralizedCUSUMState(
            is_in_learning_period=False,
            statistics={},
        )
        assert hash(state_a) == hash(state_b)


class TestGeneralizedCUSUMDimProperty:
    def test_dim_before_process_returns_initial_value(self) -> None:
        algorithm = _MinimalCUSUM()
        assert algorithm.dim == -1

    def test_dim_is_accessible_after_process(self) -> None:
        algorithm = _MinimalCUSUM()
        algorithm.process(np.array([1.0]))
        assert algorithm.dim == -1

    def test_process_sets_dim_when_dim_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        algorithm = _MinimalCUSUM()
        monkeypatch.setattr(algorithm, "_dim", None)
        algorithm.process(np.array([1.0]))
        assert algorithm.dim == 1


class TestGeneralizedCUSUMAdaptiveEstimation:
    def test_adaptive_estimation_updates_estimates(self) -> None:
        algorithm = _MinimalCUSUM(adaptive_estimation=True)
        algorithm.process(np.array([1.0]))
        algorithm.process(np.array([2.0]))

        mean_before = algorithm.estimates["mean"].copy()
        algorithm.process(np.array([3.0]))

        mean_after = algorithm.estimates["mean"]
        assert not np.allclose(mean_before, mean_after)

    def test_non_adaptive_estimation_does_not_update_estimates(self) -> None:
        algorithm = _MinimalCUSUM(adaptive_estimation=False)
        algorithm.process(np.array([1.0]))
        algorithm.process(np.array([2.0]))

        mean_before = algorithm.estimates["mean"].copy()
        algorithm.process(np.array([3.0]))

        mean_after = algorithm.estimates["mean"]
        assert np.allclose(mean_before, mean_after)


class TestGeneralizedCUSUMReset:
    def test_reset_restores_training_state(self) -> None:
        algorithm = _MinimalCUSUM()
        algorithm.process(np.array([1.0]))
        algorithm.process(np.array([2.0]))
        algorithm.process(np.array([3.0]))
        assert algorithm.state.is_in_learning_period is False

        algorithm.reset()
        assert algorithm.state.is_in_learning_period is True

    def test_reset_clears_training_buffer(self) -> None:
        algorithm = _MinimalCUSUM()
        algorithm.process(np.array([1.0]))
        algorithm.process(np.array([2.0]))
        algorithm.reset()

        first = algorithm.process(np.array([5.0]))
        second = algorithm.process(np.array([6.0]))
        assert first == 0.0
        assert second == 0.0


class TestGeneralizedCUSUMRecreate:
    def test_recreate_returns_new_instance(self) -> None:
        algorithm = _MinimalCUSUM()
        algorithm.process(np.array([1.0]))
        algorithm.process(np.array([2.0]))
        algorithm.process(np.array([3.0]))

        recreated = algorithm.recreate()

        assert recreated is not algorithm
        assert recreated.state.is_in_learning_period is True

    def test_recreate_instance_is_independent(self) -> None:
        algorithm = _MinimalCUSUM()
        algorithm.process(np.array([1.0]))
        algorithm.process(np.array([2.0]))

        recreated = algorithm.recreate()

        algorithm.process(np.array([100.0]))
        assert recreated.state.is_in_learning_period is True
