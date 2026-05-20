# -*- coding: ascii -*-
"""
Tests for crosier cusum.
"""

from __future__ import annotations

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.algorithms.online.cusum.algorithm.crosier_cusum import (
    CrosierCusum,
    CrosierCusumConfiguration,
)


def test_crosier_configuration_validates_inputs() -> None:
    with pytest.raises(ValueError, match="learning_period_size"):
        CrosierCusumConfiguration(learning_period_size=0)
    with pytest.raises(ValueError, match="cov_reg"):
        CrosierCusumConfiguration(learning_period_size=2, cov_reg=0.0)


def test_crosier_algorithm_exposes_public_properties() -> None:
    algorithm = CrosierCusum(learning_period_size=2, delta=0.3)
    first = algorithm.process(np.array([1.0]))
    second = algorithm.process(np.array([2.0]))

    assert isinstance(float(first), float)
    assert isinstance(float(second), float)
    assert algorithm.configuration.delta == 0.3
    assert "delta=0.3" in repr(algorithm)
    assert algorithm.state.statistics is not None


def test_crosier_configuration_repr_and_type() -> None:
    config = CrosierCusumConfiguration(learning_period_size=5, delta=0.7)
    assert repr(config) == "delta=0.7"
