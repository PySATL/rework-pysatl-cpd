# -*- coding: ascii -*-
"""
Tests for arm monitoring.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math

import numpy as np
import pytest

from pysatl_cpd.algorithms.online.cusum.component.estimator import GaussianARSchema
from pysatl_cpd.algorithms.online.cusum.component.monitoring import GaussianARMonitoringSchema


def _make_training_sequence() -> list[np.ndarray]:
    return [
        np.array([1.0]),
        np.array([2.2]),
        np.array([2.9]),
        np.array([4.1]),
        np.array([5.0]),
    ]


class TestGaussianARMonitoringSchema:
    def test_returns_standardized_forecast_residual(self) -> None:
        schema = GaussianARSchema(autoreg_order=2)
        schema.train(_make_training_sequence())

        monitoring = GaussianARMonitoringSchema()
        residual = monitoring.evaluate(np.array([5.8]), schema.estimates)

        assert residual.shape == (1,)
        assert math.isfinite(float(residual[0]))

    def test_rejects_multidim_observation(self) -> None:
        monitoring = GaussianARMonitoringSchema()
        estimates = {
            "intercept": 0.0,
            "coefficients": np.array([0.5]),
            "noise_variance": 1.0,
            "history": np.array([1.0]),
        }
        with pytest.raises(ValueError, match="dim=1"):
            monitoring.evaluate(np.array([1.0, 2.0]), estimates)

    def test_rejects_empty_coefficients(self) -> None:
        monitoring = GaussianARMonitoringSchema()
        estimates = {
            "intercept": 0.0,
            "coefficients": np.array([]),
            "noise_variance": 1.0,
            "history": np.array([1.0]),
        }
        with pytest.raises(ValueError, match="fitted"):
            monitoring.evaluate(np.array([1.0]), estimates)

    def test_reset_does_not_error(self) -> None:
        monitoring = GaussianARMonitoringSchema()
        result = monitoring.reset()
        assert result is None
