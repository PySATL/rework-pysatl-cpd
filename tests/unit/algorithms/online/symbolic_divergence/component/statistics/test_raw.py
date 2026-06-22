# -*- coding: ascii -*-
"""
Tests for raw divergence statistic.
"""

from __future__ import annotations

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.algorithms.online.symbolic_divergence.statistics.raw import RawDivergenceStatistic


class TestRawDivergenceStatistic:
    def test_initial_value_is_zero(self) -> None:
        assert RawDivergenceStatistic().value == 0.0

    def test_update_stores_latest_divergence(self) -> None:
        statistic = RawDivergenceStatistic()
        statistic.update(0.7, sample_size=10)
        assert statistic.value == 0.7

    def test_sample_size_is_ignored(self) -> None:
        statistic = RawDivergenceStatistic()
        statistic.update(0.4, sample_size=1)
        first = statistic.value
        statistic.update(0.4, sample_size=1000)
        assert statistic.value == first

    def test_reset_clears_value(self) -> None:
        statistic = RawDivergenceStatistic()
        statistic.update(0.9, sample_size=5)
        statistic.reset()
        assert statistic.value == 0.0
