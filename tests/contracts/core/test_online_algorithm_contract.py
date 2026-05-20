# -*- coding: ascii -*-
"""
Tests for online algorithm contract.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.core.online import OnlineAlgorithm
from tests.support.core import assert_nested_equal


def _normalize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    return value


def _assert_state_equal(actual, expected) -> None:
    assert type(actual) is type(expected)
    assert_nested_equal(_normalize(actual.__dict__), _normalize(expected.__dict__))


class OnlineAlgorithmContract:
    """Reusable checks for OnlineAlgorithm implementations."""

    @pytest.fixture
    def algorithm(self) -> OnlineAlgorithm:
        raise NotImplementedError

    @pytest.fixture
    def sample_observation(self):
        raise NotImplementedError

    @pytest.fixture
    def update_observations(self, sample_observation):
        return [sample_observation]

    @pytest.fixture
    def fresh_state(self):
        raise NotImplementedError

    def test_name_is_non_empty_string(self, algorithm: OnlineAlgorithm) -> None:
        assert isinstance(algorithm.name, str)
        assert algorithm.name

    def test_configuration_has_expected_type(self, algorithm: OnlineAlgorithm) -> None:
        assert isinstance(algorithm.configuration, type(algorithm.configuration))

    def test_state_has_expected_type(self, algorithm: OnlineAlgorithm) -> None:
        assert isinstance(algorithm.state, type(algorithm.state))

    def test_description_combines_name_and_configuration(self, algorithm: OnlineAlgorithm) -> None:
        description = algorithm.description
        assert description.name == algorithm.name
        assert description.configuration == algorithm.configuration

    def test_process_returns_number(self, algorithm: OnlineAlgorithm, sample_observation) -> None:
        assert isinstance(algorithm.process(sample_observation), int | float)

    def test_process_updates_state(self, algorithm: OnlineAlgorithm, update_observations) -> None:
        before = algorithm.state
        for observation in update_observations:
            algorithm.process(observation)
        assert _normalize(algorithm.state.__dict__) != _normalize(before.__dict__)

    def test_reset_restores_fresh_state(self, algorithm: OnlineAlgorithm, update_observations, fresh_state) -> None:
        for observation in update_observations:
            algorithm.process(observation)
        algorithm.reset()
        _assert_state_equal(algorithm.state, fresh_state)

    def test_recreate_returns_same_concrete_type(self, algorithm: OnlineAlgorithm) -> None:
        assert type(algorithm.recreate()) is type(algorithm)

    def test_recreate_preserves_configuration_and_returns_fresh_state(
        self, algorithm: OnlineAlgorithm, fresh_state
    ) -> None:
        recreated = algorithm.recreate()
        assert recreated is not algorithm
        assert recreated.configuration == algorithm.configuration
        _assert_state_equal(recreated.state, fresh_state)
