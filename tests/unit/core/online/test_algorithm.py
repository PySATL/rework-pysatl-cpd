# -*- coding: ascii -*-
"""
Tests for algorithm.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.core.online.ionline_algorithm import (
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmDescription,
    OnlineAlgorithmState,
)
from pysatl_cpd.typedefs import stable_hash
from tests.support.core import SimpleAlgorithm, SimpleAlgorithmConfig, SimpleAlgorithmState


class TestOnlineAlgorithmState:
    """Tests for OnlineAlgorithmState."""

    def test_default_is_in_learning_period(self) -> None:
        """Default is_in_learning_period should be False."""
        state = OnlineAlgorithmState()
        assert not state.is_in_learning_period

    def test_custom_is_in_learning_period(self) -> None:
        """is_in_learning_period should be settable."""
        state = OnlineAlgorithmState(is_in_learning_period=True)
        assert state.is_in_learning_period

    def test_hash_uses_stable_hash(self) -> None:
        state = OnlineAlgorithmState(is_in_learning_period=True)
        assert hash(state) == stable_hash((state.__class__.__module__, state.__class__.__qualname__, True))


class TestOnlineAlgorithmConfiguration:
    """Tests for OnlineAlgorithmConfiguration."""

    def test_default_learning_period_size(self) -> None:
        """Default learning_period_size should be 0."""
        config = OnlineAlgorithmConfiguration()
        assert config.learning_period_size == 0

    def test_custom_learning_period_size(self) -> None:
        """learning_period_size should be settable."""
        config = OnlineAlgorithmConfiguration(learning_period_size=10)
        assert config.learning_period_size == 10

    def test_hash_uses_stable_hash(self) -> None:
        config = OnlineAlgorithmConfiguration(learning_period_size=10)
        assert hash(config) == stable_hash((config.__class__.__module__, config.__class__.__qualname__, 10))

    def test_concrete_configuration_hash_uses_stable_hash(self) -> None:
        config = SimpleAlgorithmConfig(learning_period_size=10, threshold=0.7)
        assert hash(config) == stable_hash((config.__class__.__module__, config.__class__.__qualname__, 10, 0.7))


class TestOnlineAlgorithmDescription:
    """Tests for OnlineAlgorithmDescription."""

    def test_creation(self) -> None:
        """Should store name and configuration."""
        config = SimpleAlgorithmConfig(threshold=0.7)
        desc = OnlineAlgorithmDescription(name="TestAlgo", configuration=config)
        assert desc.name == "TestAlgo"
        assert desc.configuration == config

    def test_hash_stable(self) -> None:
        """Hash should be stable for same name and config."""
        config = SimpleAlgorithmConfig(threshold=0.5)
        desc1 = OnlineAlgorithmDescription(name="Algo", configuration=config)
        desc2 = OnlineAlgorithmDescription(name="Algo", configuration=config)
        assert hash(desc1) == hash(desc2)

    def test_hash_uses_stable_hash(self) -> None:
        config = SimpleAlgorithmConfig(learning_period_size=3, threshold=0.5)
        desc = OnlineAlgorithmDescription(name="Algo", configuration=config)
        assert hash(desc) == stable_hash((desc.__class__.__module__, desc.__class__.__qualname__, "Algo", config))

    def test_frozen(self) -> None:
        """Description should be frozen (immutable)."""
        config = SimpleAlgorithmConfig()
        desc = OnlineAlgorithmDescription(name="Algo", configuration=config)
        try:
            desc.name = "New"  # type: ignore
            raise AssertionError("Should raise AttributeError")
        except AttributeError:
            pass


class TestOnlineAlgorithm:
    """Tests for OnlineAlgorithm abstract source class behavior."""

    def test_process_returns_number(self) -> None:
        """process() should return a Number."""
        algo = SimpleAlgorithm(SimpleAlgorithmConfig())
        result = algo.process(1.0)
        assert isinstance(result, int | float)

    def test_reset_changes_state(self) -> None:
        """reset() should reset the algorithm state."""
        algo = SimpleAlgorithm(SimpleAlgorithmConfig())
        algo.process(1.0)
        algo.reset()
        assert algo.state == SimpleAlgorithmState()

    def test_recreate_returns_instance(self) -> None:
        """recreate() should return an instance of the same class."""
        config = SimpleAlgorithmConfig(threshold=0.8)
        result = SimpleAlgorithm.recreate(config)
        assert isinstance(result, SimpleAlgorithm)

    def test_recreate_with_state(self) -> None:
        """recreate() should restore the given state."""
        config = SimpleAlgorithmConfig()
        state = SimpleAlgorithmState(value=99.0)
        algo = SimpleAlgorithm.recreate(config, state)
        assert algo.state == state

    def test_description_property(self) -> None:
        """description property should return OnlineAlgorithmDescription."""
        algo = SimpleAlgorithm(SimpleAlgorithmConfig(threshold=0.6))
        desc = algo.description
        assert isinstance(desc, OnlineAlgorithmDescription)
        assert desc.name == "SimpleAlgorithm"
        assert desc.configuration.threshold == 0.6

    def test_repr(self) -> None:
        """__repr__ should include name and configuration."""
        algo = SimpleAlgorithm(SimpleAlgorithmConfig(threshold=0.5))
        repr_str = repr(algo)
        assert "SimpleAlgorithm" in repr_str
        assert "threshold=0.5" in repr_str

    def test_name_property(self) -> None:
        """name property should return class name."""
        algo = SimpleAlgorithm(SimpleAlgorithmConfig())
        assert algo.name == "SimpleAlgorithm"
