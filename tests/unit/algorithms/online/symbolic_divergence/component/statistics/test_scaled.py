# -*- coding: ascii -*-
"""
Tests for scaled divergence statistic.
"""

from __future__ import annotations

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.scaled import ScaledDivergenceStatistic


class TestScaledDivergenceStatisticConstruction:
    def test_rejects_negative_scale(self) -> None:
        with pytest.raises(ValueError, match="scale must be non-negative"):
            ScaledDivergenceStatistic(scale=-1.0)


class TestScaledDivergenceStatistic:
    def test_initial_value_is_zero(self) -> None:
        assert ScaledDivergenceStatistic().value == 0.0

    def test_default_scale_produces_two_n_d(self) -> None:
        statistic = ScaledDivergenceStatistic()
        statistic.update(0.5, sample_size=8)
        assert statistic.value == pytest.approx(2.0 * 8 * 0.5)

    def test_custom_scale_is_applied(self) -> None:
        statistic = ScaledDivergenceStatistic(scale=3.0)
        statistic.update(0.25, sample_size=4)
        assert statistic.value == pytest.approx(3.0 * 4 * 0.25)

    def test_reset_clears_value(self) -> None:
        statistic = ScaledDivergenceStatistic()
        statistic.update(0.5, sample_size=8)
        statistic.reset()
        assert statistic.value == 0.0
