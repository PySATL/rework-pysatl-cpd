# -*- coding: ascii -*-
"""
Tests for autoregressive cusum.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math

import numpy as np
import pytest

from pysatl_cpd.algorithms.online import AutoregressiveCUSUM
from pysatl_cpd.algorithms.online.cusum.algorithm.autoregressive_cusum import AutoregressiveCusumConfiguration


def _make_training_sequence() -> list[np.ndarray]:
    return [
        np.array([1.0]),
        np.array([2.2]),
        np.array([2.9]),
        np.array([4.1]),
        np.array([5.0]),
    ]


class TestAutoregressiveCusumConfiguration:
    def test_rejects_non_positive_learning_period_size(self) -> None:
        with pytest.raises(ValueError, match="learning_period_size"):
            AutoregressiveCusumConfiguration(learning_period_size=0, autoreg_order=1)

    def test_rejects_non_positive_autoreg_order(self) -> None:
        with pytest.raises(ValueError, match="autoreg_order"):
            AutoregressiveCusumConfiguration(learning_period_size=5, autoreg_order=0)

    def test_rejects_learning_period_too_small_for_order(self) -> None:
        with pytest.raises(ValueError, match="learning_period_size must be at least"):
            AutoregressiveCusumConfiguration(learning_period_size=2, autoreg_order=2)

    def test_rejects_autoreg_window_not_greater_than_order(self) -> None:
        with pytest.raises(ValueError, match="autoreg_window"):
            AutoregressiveCusumConfiguration(learning_period_size=5, autoreg_order=2, autoreg_window=1)

    def test_identical_configs_have_same_hash(self) -> None:
        config_a = AutoregressiveCusumConfiguration(
            learning_period_size=10,
            delta=0.5,
            autoreg_order=3,
            autoreg_window=20,
        )
        config_b = AutoregressiveCusumConfiguration(
            learning_period_size=10,
            delta=0.5,
            autoreg_order=3,
            autoreg_window=20,
        )
        assert hash(config_a) == hash(config_b)
        assert config_a == config_b


class TestAutoregressiveCUSUM:
    def test_learning_period_emits_zero_and_then_scores(self) -> None:
        algorithm = AutoregressiveCUSUM(learning_period_size=5, delta=0.0, autoreg_order=2)

        first = algorithm.process(1.0)
        second = algorithm.process(np.array([2.2]))
        third = algorithm.process(2.9)
        fourth = algorithm.process(np.array([4.1]))
        fifth = algorithm.process(5.0)
        sixth = algorithm.process(5.8)

        assert first == 0.0
        assert second == 0.0
        assert third == 0.0
        assert fourth == 0.0
        assert fifth == 0.0
        assert math.isfinite(float(sixth))

    def test_rejects_non_univariate_observations(self) -> None:
        algorithm = AutoregressiveCUSUM(learning_period_size=5, delta=0.0, autoreg_order=2)

        with pytest.raises(ValueError, match="dim=1"):
            algorithm.process(np.array([1.0, 2.0]))

    def test_state_contains_weights_not_model(self) -> None:
        algorithm = AutoregressiveCUSUM(learning_period_size=5, delta=0.0, autoreg_order=2)
        for observation in [1.0, 2.2, 2.9, 4.1, 5.0]:
            algorithm.process(observation)

        state = algorithm.state
        assert set(state.statistics) == {"intercept", "coefficients", "noise_variance", "history"}

    def test_configuration_returns_autoregressive_cusum_configuration(self) -> None:
        algorithm = AutoregressiveCUSUM(learning_period_size=5, delta=0.0, autoreg_order=2)
        assert isinstance(algorithm.configuration, AutoregressiveCusumConfiguration)
