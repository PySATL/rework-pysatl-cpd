# -*- coding: ascii -*-
"""
Tests for max run length.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.algorithms.online.bayesian.component.cpf import MaxRunLengthCPF


class TestMaxRunLengthCPF:
    def test_uses_complement_of_last_probability(self) -> None:
        cpf = MaxRunLengthCPF()
        score = cpf.calculate(np.log(np.array([0.2, 0.8], dtype=np.float64)))
        assert score == pytest.approx(0.2)

    def test_with_empty_array_returns_zero(self) -> None:
        cpf = MaxRunLengthCPF()
        assert cpf.calculate(np.array([], dtype=np.float64)) == 0.0
