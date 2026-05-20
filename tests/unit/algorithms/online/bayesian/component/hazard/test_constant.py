# -*- coding: ascii -*-
"""
Tests for constant.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math

import numpy as np
import pytest

from pysatl_cpd.algorithms.online.bayesian.component.hazard import ConstantHazard


class TestConstantHazard:
    def test_returns_constant_log_hazard_values(self) -> None:
        hazard = ConstantHazard(lambda_=5.0)

        log_h, log_neg_h = hazard.hazard(np.array([0, 1, 2], dtype=np.intp))

        assert log_h.shape == (3,)
        assert log_neg_h.shape == (3,)
        assert np.allclose(log_h, -math.log(5.0))
        assert np.allclose(log_neg_h, math.log(1.0 - 1.0 / 5.0))

    def test_validation_raises_value_error_when_lambda_below_one(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            ConstantHazard(lambda_=0.5)
        assert "lambda_" in str(exc_info.value)
