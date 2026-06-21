# -*- coding: ascii -*-
"""
Tests for kl divergence.
"""

from __future__ import annotations

__author__ = "Yulia Burmistrova"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.algorithms.online.symbolic_divergence.divergences.kl_divergence import KLDivergence


class TestKLDivergenceConstruction:
    def test_rejects_negative_smoothing(self) -> None:
        with pytest.raises(ValueError, match="smoothing must be non-negative"):
            KLDivergence(smoothing=-1e-3)


class TestKLDivergenceCompute:
    def test_identical_distributions_have_zero_divergence(self) -> None:
        divergence = KLDivergence()
        reference = np.array([1.0, 2.0, 3.0, 4.0])
        assert divergence.compute(reference, reference) == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions_have_positive_divergence(self) -> None:
        divergence = KLDivergence()
        empirical = np.array([10.0, 1.0, 1.0])
        reference = np.array([1.0, 1.0, 10.0])
        assert divergence.compute(empirical, reference) > 0.0

    def test_counts_and_probabilities_give_same_result(self) -> None:
        divergence = KLDivergence(smoothing=0.0)
        counts_empirical = np.array([2.0, 6.0, 2.0])
        counts_reference = np.array([5.0, 5.0, 10.0])
        prob_empirical = counts_empirical / counts_empirical.sum()
        prob_reference = counts_reference / counts_reference.sum()
        assert divergence.compute(counts_empirical, counts_reference) == pytest.approx(
            divergence.compute(prob_empirical, prob_reference)
        )

    def test_zero_empirical_entries_are_skipped(self) -> None:
        divergence = KLDivergence()
        empirical = np.array([0.0, 1.0, 0.0])
        reference = np.array([1.0, 1.0, 1.0])
        assert np.isfinite(divergence.compute(empirical, reference))

    def test_all_zero_empirical_returns_zero(self) -> None:
        divergence = KLDivergence()
        empirical = np.array([0.0, 0.0, 0.0])
        reference = np.array([1.0, 1.0, 1.0])
        assert divergence.compute(empirical, reference) == 0.0

    def test_zero_reference_does_not_raise_with_smoothing(self) -> None:
        divergence = KLDivergence(smoothing=1e-9)
        empirical = np.array([1.0, 1.0])
        reference = np.array([1.0, 0.0])
        assert np.isfinite(divergence.compute(empirical, reference))
