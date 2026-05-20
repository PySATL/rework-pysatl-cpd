# -*- coding: ascii -*-
"""
Tests for page.
"""

from __future__ import annotations

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.algorithms.online.cusum.component.cpf.page import ChangepointFuncUnivariatePageCUSUM


class TestChangepointFuncUnivariatePageCUSUM:
    def test_value_is_zero_when_no_observations(self) -> None:
        cpf = ChangepointFuncUnivariatePageCUSUM()
        assert cpf.value == 0.0

    def test_reset_clears_statistics(self) -> None:
        cpf = ChangepointFuncUnivariatePageCUSUM(delta=0.1)
        cpf.update(np.array([1.0]))
        assert cpf.value > 0.0
        cpf.reset()
        assert cpf.value == 0.0

    def test_pos_side_only_ignores_negative_observations(self) -> None:
        cpf = ChangepointFuncUnivariatePageCUSUM(delta=0.0, side="pos")
        cpf.update(np.array([-1.0]))
        assert cpf.value == 0.0
        cpf.update(np.array([1.0]))
        assert cpf.value > 0.0

    def test_neg_side_only_ignores_positive_observations(self) -> None:
        cpf = ChangepointFuncUnivariatePageCUSUM(delta=0.0, side="neg")
        cpf.update(np.array([1.0]))
        assert cpf.value == 0.0
        cpf.update(np.array([-1.0]))
        assert cpf.value > 0.0

    def test_shape_mismatch_raises_value_error(self) -> None:
        cpf = ChangepointFuncUnivariatePageCUSUM()
        with pytest.raises(ValueError, match="shape mismatch"):
            cpf.update(np.array([1.0, 2.0]))
