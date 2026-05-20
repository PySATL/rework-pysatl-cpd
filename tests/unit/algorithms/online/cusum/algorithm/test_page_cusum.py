# -*- coding: ascii -*-
"""
Tests for page cusum.
"""

from __future__ import annotations

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.algorithms.online.cusum.algorithm.page_cusum import (
    PageTwoSidedCusum,
    PageTwoSidedCusumConfiguration,
)


def test_page_configuration_validates_inputs() -> None:
    with pytest.raises(ValueError, match="learning_period_size"):
        PageTwoSidedCusumConfiguration(learning_period_size=0)
    with pytest.raises(ValueError, match="cov_reg"):
        PageTwoSidedCusumConfiguration(learning_period_size=2, cov_reg=0.0)


def test_page_algorithm_exposes_public_properties() -> None:
    algorithm = PageTwoSidedCusum(learning_period_size=2, delta=0.4)
    first = algorithm.process(np.array([1.0]))
    second = algorithm.process(np.array([2.0]))

    assert isinstance(float(first), float)
    assert isinstance(float(second), float)
    assert algorithm.configuration.delta == 0.4
    assert "delta=0.4" in repr(algorithm)
    assert algorithm.state.statistics is not None


def test_page_configuration_repr_and_type() -> None:
    config = PageTwoSidedCusumConfiguration(learning_period_size=5, delta=0.8)
    assert repr(config) == "delta=0.8"
