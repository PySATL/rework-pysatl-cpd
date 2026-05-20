# -*- coding: ascii -*-
"""
Tests for variance cusum.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math

import numpy as np
import pytest

from pysatl_cpd.algorithms.online import VarianceTwoSidedCUSUM
from pysatl_cpd.algorithms.online.cusum.algorithm.variance_cusum import VarianceTwoSidedCusumConfiguration


class TestVarianceTwoSidedCusumConfiguration:
    def test_rejects_non_positive_learning_period_size(self) -> None:
        with pytest.raises(ValueError, match="learning_period_size"):
            VarianceTwoSidedCusumConfiguration(learning_period_size=0)

    def test_identical_configs_have_same_hash(self) -> None:
        config_a = VarianceTwoSidedCusumConfiguration(learning_period_size=10, delta=0.5)
        config_b = VarianceTwoSidedCusumConfiguration(learning_period_size=10, delta=0.5)
        assert hash(config_a) == hash(config_b)
        assert config_a == config_b


class TestVarianceTwoSidedCUSUM:
    def test_learning_period_emits_zero_and_then_scores(self) -> None:
        algorithm = VarianceTwoSidedCUSUM(learning_period_size=2, delta=0.0)

        first = algorithm.process(10.0)
        second = algorithm.process(np.array([11.0]))
        third = algorithm.process(13.0)

        assert first == 0.0
        assert second == 0.0
        assert math.isfinite(float(third))
        assert algorithm.state.is_in_learning_period is False

    def test_rejects_non_univariate_observations(self) -> None:
        algorithm = VarianceTwoSidedCUSUM(learning_period_size=2)

        with pytest.raises(ValueError, match="dim=1"):
            algorithm.process(np.array([1.0, 2.0]))

    def test_configuration_returns_variance_two_sided_cusum_configuration(self) -> None:
        algorithm = VarianceTwoSidedCUSUM(learning_period_size=2, delta=0.0)
        assert isinstance(algorithm.configuration, VarianceTwoSidedCusumConfiguration)
