# -*- coding: ascii -*-
"""
Tests for gaussian mle.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np

from pysatl_cpd.algorithms.online.cusum.component.estimator import GaussianMLESchema


class TestGaussianMLESchema:
    def test_non_adaptive_update_does_nothing(self) -> None:
        schema = GaussianMLESchema(adaptive=False)
        schema.train([np.array([1.0]), np.array([2.0]), np.array([3.0])])
        cov_before = schema.cov.copy()
        mean_before = schema.mean.copy()

        schema.update(np.array([100.0]))
        np.testing.assert_array_equal(schema.cov, cov_before)
        np.testing.assert_array_equal(schema.mean, mean_before)

    def test_reset_clears_accumulators(self) -> None:
        schema = GaussianMLESchema(adaptive=True)
        schema.train([np.array([1.0]), np.array([2.0]), np.array([3.0])])
        assert schema._dim != -1

        schema.reset()
        assert schema._len == 0
        assert schema._dim == -1
        assert schema._welford_mean.shape == (0,)
        assert schema._welford_m2.shape == (0, 0)
