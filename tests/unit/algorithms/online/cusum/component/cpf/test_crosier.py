# -*- coding: ascii -*-
"""
Tests for crosier.
"""

from __future__ import annotations

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.algorithms.online.cusum.component.cpf.crosier import ChangepointFuncCrosierCUSUM


class TestChangepointFuncCrosierCUSUM:
    def test_update_sets_dim_from_first_observation(self) -> None:
        cpf = ChangepointFuncCrosierCUSUM(dim=2, delta=0.5)
        assert cpf.dim == -1
        cpf.update(np.array([1.0, 2.0]))
        assert cpf.dim == 2

    def test_value_returns_norm_of_stat(self) -> None:
        cpf = ChangepointFuncCrosierCUSUM(dim=2, delta=0.0)
        cpf.update(np.array([3.0, 4.0]))
        expected_norm = float(np.linalg.norm(np.array([3.0, 4.0])))
        assert cpf.value == pytest.approx(expected_norm)

    def test_reset_clears_dim_and_stat(self) -> None:
        cpf = ChangepointFuncCrosierCUSUM(dim=2)
        cpf.update(np.array([1.0, 2.0]))
        assert cpf.dim != -1
        cpf.reset()
        assert cpf.dim == -1
        assert cpf.stat.shape == (0,)

    def test_second_update_uses_existing_dim(self) -> None:
        cpf = ChangepointFuncCrosierCUSUM(dim=2, delta=0.5)
        cpf.update(np.array([1.0, 2.0]))
        cpf.update(np.array([0.5, 1.0]))
        assert cpf.dim == 2
        assert np.isfinite(cpf.value)
