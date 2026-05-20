"""
Tests for gaussian.
"""

from __future__ import annotations

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.algorithms.online.cusum.component.monitoring.gaussian import GaussianMonitoringSchema


def _schema_with_dim(dim: int, cov_reg: float = 1e-6) -> GaussianMonitoringSchema:
    """Create a schema with dim already set (bypassing evaluate)."""
    schema = GaussianMonitoringSchema(cov_reg=cov_reg)
    schema.dim = dim
    return schema


class TestGaussianMonitoringSchema:
    def test_constructor_sets_defaults(self) -> None:
        schema = GaussianMonitoringSchema(cov_reg=1e-6)
        assert schema.dim == -1
        assert schema.cov_reg == 1e-6

    def test_evaluate_sets_dim_from_first_observation(self) -> None:
        schema = GaussianMonitoringSchema(cov_reg=1e-6)
        assert schema.dim == -1

        obs = np.array([1.0, 2.0, 3.0])
        params: dict = {"mean": np.zeros(3), "cov": np.eye(3)}
        schema.evaluate(obs, params)

        assert schema.dim == 3

    def test_evaluate_does_not_change_dim_on_subsequent_calls(self) -> None:
        schema = GaussianMonitoringSchema(cov_reg=1e-6)
        obs1 = np.array([1.0, 2.0])
        obs2 = np.array([3.0, 4.0])
        params: dict = {"mean": np.zeros(2), "cov": np.eye(2)}

        schema.evaluate(obs1, params)
        assert schema.dim == 2

        schema.evaluate(obs2, params)
        assert schema.dim == 2

    def test_reset_clears_dim(self) -> None:
        schema = GaussianMonitoringSchema(cov_reg=1e-6)
        obs = np.array([1.0, 2.0])
        params: dict = {"mean": np.zeros(2), "cov": np.eye(2)}

        schema.evaluate(obs, params)
        assert schema.dim == 2

        schema.reset()
        assert schema.dim == -1

    def test_evaluate_returns_whitened_residual(self) -> None:
        """For cov=I and mean=0, whitened residual equals observation."""
        schema = GaussianMonitoringSchema(cov_reg=1e-6)
        obs = np.array([0.5, -0.3])
        params: dict = {"mean": np.zeros(2), "cov": np.eye(2)}

        residual = schema.evaluate(obs, params)
        expected = np.array([0.5, -0.3])
        np.testing.assert_array_almost_equal(residual, expected)

    def test_evaluate_with_non_identity_covariance(self) -> None:
        """Whitening with non-identity covariance."""
        schema = GaussianMonitoringSchema(cov_reg=1e-6)
        obs = np.array([1.0, 0.0])
        # cov = [[2, 0], [0, 1]] => inv_sqrt = [[1/sqrt(2), 0], [0, 1]]
        params: dict = {"mean": np.zeros(2), "cov": np.array([[2.0, 0.0], [0.0, 1.0]])}

        residual = schema.evaluate(obs, params)
        # expected: inv_sqrt @ obs = [1/sqrt(2), 0]
        expected = np.array([1.0 / np.sqrt(2.0), 0.0])
        np.testing.assert_array_almost_equal(residual, expected)

    def test_evaluate_with_scalar_observation(self) -> None:
        """Scalar observation is coerced to 1-D array."""
        schema = GaussianMonitoringSchema(cov_reg=1e-6)
        obs = 0.5
        params: dict = {"mean": np.zeros(1), "cov": np.eye(1)}

        residual = schema.evaluate(obs, params)
        np.testing.assert_array_almost_equal(residual, np.array([0.5]))

    def test_inv_mat_sqrt_identity(self) -> None:
        """_inv_mat_sqrt of identity returns identity."""
        schema = _schema_with_dim(dim=3, cov_reg=0.0)
        result = schema._inv_mat_sqrt(np.eye(3))
        np.testing.assert_array_almost_equal(result, np.eye(3))

    @pytest.mark.parametrize("cov_reg", [0.0, 1e-12, 1e-6])
    def test_inv_mat_sqrt_symmetric_result(self, cov_reg: float) -> None:
        """_inv_mat_sqrt always returns a symmetric matrix."""
        schema = _schema_with_dim(dim=2, cov_reg=cov_reg)
        mat = np.array([[2.0, 0.5], [0.5, 1.0]])
        result = schema._inv_mat_sqrt(mat)
        np.testing.assert_array_almost_equal(result, result.T)

    def test_inv_mat_sqrt_nearly_singular(self) -> None:
        """Near-singular matrix with regularization."""
        schema = _schema_with_dim(dim=3, cov_reg=1e-6)
        mat = np.zeros((3, 3))
        result = schema._inv_mat_sqrt(mat)
        assert np.all(np.isfinite(result))
        np.testing.assert_array_almost_equal(result, result.T)

    def test_evaluate_with_dim_already_set(self) -> None:
        """Second evaluate call when dim is already set."""
        schema = GaussianMonitoringSchema(cov_reg=1e-6)
        estimates = {"mean": np.array([0.0]), "cov": np.eye(1)}
        schema.evaluate(np.array([1.0]), estimates)
        # Second call: dim already == 1, so the if-self.dim==-1 branch is skipped
        result = schema.evaluate(np.array([2.0]), estimates)
        assert result.shape == (1,)

    def test_inv_mat_sqrt_returns_correct_shape(self) -> None:
        """_inv_mat_sqrt returns the correct shape and inverts the matrix."""
        schema = GaussianMonitoringSchema(cov_reg=1e-6)
        schema.dim = 2
        mat = np.array([[4.0, 0.0], [0.0, 9.0]])
        result = schema._inv_mat_sqrt(mat)
        assert result.shape == (2, 2)
        # Verify: result @ (mat + cov_reg*I) @ result ~= I
        # since _inv_mat_sqrt internally regularises with cov_reg
        regularised = 0.5 * (mat + mat.T) + (1e-6) * np.eye(2)
        reconstructed = result @ regularised @ result
        np.testing.assert_array_almost_equal(reconstructed, np.eye(2), decimal=10)
