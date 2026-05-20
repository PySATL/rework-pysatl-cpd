# -*- coding: ascii -*-
"""
Tests for drop.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.algorithms.online.bayesian.component.cpf import DropCPF


class TestDropCPF:
    def test_tracks_positive_drop_between_steps(self) -> None:
        cpf = DropCPF()

        first = cpf.calculate(np.log(np.array([0.2, 0.8], dtype=np.float64)))
        second = cpf.calculate(np.log(np.array([0.5, 0.5], dtype=np.float64)))
        cpf.clear()
        third = cpf.calculate(np.log(np.array([0.5, 0.5], dtype=np.float64)))

        assert first == 0.0
        assert second == pytest.approx(0.3)
        assert third == 0.0

    def test_init_has_none_previous_log_prob(self) -> None:
        cpf = DropCPF()
        assert cpf._previous_log_prob_max_run is None

    def test_with_empty_array_returns_zero(self) -> None:
        cpf = DropCPF()
        assert cpf.calculate(np.array([], dtype=np.float64)) == 0.0
