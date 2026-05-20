# -*- coding: ascii -*-
"""
Tests for gaussian ar.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math

import numpy as np
import pytest

from pysatl_cpd.algorithms.online.cusum.component.estimator import GaussianARSchema


def _make_training_sequence() -> list[np.ndarray]:
    return [
        np.array([1.0]),
        np.array([2.2]),
        np.array([2.9]),
        np.array([4.1]),
        np.array([5.0]),
    ]


class TestGaussianARSchema:
    def test_raises_clear_error_when_arch_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pysatl_cpd.algorithms.online.cusum.component.estimator.gaussian_ar as gaussian_ar

        monkeypatch.setattr(gaussian_ar, "HAS_ARCH", False)

        with pytest.raises(RuntimeError, match="optional 'arch' dependency"):
            gaussian_ar.GaussianARSchema()

    def test_trains_and_exposes_weights_not_model(self) -> None:
        schema = GaussianARSchema(autoreg_order=2)
        schema.train(_make_training_sequence())

        estimates = schema.estimates

        assert set(estimates) == {"intercept", "coefficients", "noise_variance", "history"}
        assert estimates["coefficients"].shape == (2,)
        assert estimates["history"].shape == (2,)
        assert math.isfinite(estimates["intercept"])
        assert math.isfinite(estimates["noise_variance"])

    def test_update_refits_weights_and_tracks_recent_history(self) -> None:
        schema = GaussianARSchema(autoreg_order=2, window=5)
        schema.train(_make_training_sequence())
        before = schema.estimates

        schema.update(np.array([5.8]))
        after = schema.estimates

        assert after["history"].tolist() == pytest.approx([5.0, 5.8])
        assert not np.allclose(before["coefficients"], after["coefficients"])

    def test_raises_on_zero_autoreg_order(self) -> None:
        with pytest.raises(ValueError, match="autoreg_order"):
            GaussianARSchema(autoreg_order=0)

    def test_raises_on_zero_window(self) -> None:
        with pytest.raises(ValueError, match="window"):
            GaussianARSchema(window=0)

    def test_coerce_scalar_rejects_multidim(self) -> None:
        schema = GaussianARSchema(autoreg_order=1)
        with pytest.raises(ValueError, match="dim=1"):
            schema._coerce_scalar(np.array([1.0, 2.0]))

    def test_train_raises_with_too_few_observations(self) -> None:
        schema = GaussianARSchema(autoreg_order=3)
        with pytest.raises(ValueError, match="Not enough observations"):
            schema.train([np.array([1.0]), np.array([2.0])])

    def test_non_adaptive_update_does_not_refit(self) -> None:
        schema = GaussianARSchema(autoreg_order=2, adaptive=False)
        schema.train(_make_training_sequence())
        before = schema.estimates

        schema.update(np.array([5.8]))
        after = schema.estimates

        np.testing.assert_array_equal(before["coefficients"], after["coefficients"])
        np.testing.assert_array_equal(before["history"], after["history"])

    def test_reset_clears_all_state(self) -> None:
        schema = GaussianARSchema(autoreg_order=2)
        schema.train(_make_training_sequence())
        assert len(schema.estimates["coefficients"]) > 0

        schema.reset()
        estimates = schema.estimates
        assert estimates["intercept"] == 0.0
        assert estimates["coefficients"].shape == (0,)
        assert estimates["noise_variance"] == 0.0
        assert estimates["history"].shape == (0,)
