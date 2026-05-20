# -*- coding: ascii -*-
"""
Tests for variance.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math

import numpy as np
import pytest

from pysatl_cpd.algorithms.online.cusum.component.monitoring.variance import VarianceMonitoringSchema


class TestVarianceMonitoringSchema:
    def test_first_step_uses_estimated_mean_as_previous_observation(self) -> None:
        schema = VarianceMonitoringSchema()

        residual = schema.evaluate(
            np.array([12.0]),
            {"mean": np.array([10.0]), "cov": np.array([[4.0]])},
        )

        expected = (math.sqrt(abs((abs((12.0 - 10.0) / math.sqrt(2.0))) / math.sqrt(4.0))) - 0.82218) / 0.34914
        assert residual.shape == (1,)
        assert residual[0] == pytest.approx(expected)

    def test_reset_clears_previous_observation(self) -> None:
        schema = VarianceMonitoringSchema()
        estimates = {"mean": np.array([10.0]), "cov": np.array([[4.0]])}

        first = schema.evaluate(np.array([12.0]), estimates)
        schema.reset()
        second = schema.evaluate(np.array([12.0]), estimates)

        assert second[0] == pytest.approx(first[0])

    def test_evaluate_rejects_multidim_observation(self) -> None:
        schema = VarianceMonitoringSchema()
        estimates = {"mean": np.array([10.0]), "cov": np.array([[4.0]])}
        with pytest.raises(ValueError, match="dim=1"):
            schema.evaluate(np.array([1.0, 2.0]), estimates)

    def test_second_evaluate_uses_stored_previous_observation(self) -> None:
        schema = VarianceMonitoringSchema()
        estimates = {"mean": np.array([10.0]), "cov": np.array([[4.0]])}
        schema.evaluate(np.array([12.0]), estimates)
        result = schema.evaluate(np.array([14.0]), estimates)
        assert result.shape == (1,)
        assert np.isfinite(float(result[0]))
