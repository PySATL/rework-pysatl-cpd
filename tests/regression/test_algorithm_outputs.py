# -*- coding: ascii -*-
"""
Tests for algorithm outputs.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pathlib import Path

import numpy as np
import pytest

from pysatl_cpd.algorithms.online import (
    AutoregressiveCUSUM,
    CrosierCusum,
    PageTwoSidedCusum,
    ShewhartControlChart,
    SlopeKLSymbolicDivergence,
    UnivariateGaussianConjugateBOCPD,
    VarianceTwoSidedCUSUM,
)
from pysatl_cpd.algorithms.online.bayesian import BayesianCPFType
from tests.support.golden import load_json_golden


def _normalize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    return value


def _assert_nested_close(actual, expected) -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, expected_value in expected.items():
            _assert_nested_close(actual[key], expected_value)
        return
    if isinstance(expected, list):
        assert np.allclose(np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64))
        return
    if isinstance(expected, float):
        assert float(actual) == pytest.approx(expected)
        return
    assert actual == expected


def _state_snapshot(state, golden_state: dict) -> dict:
    return {key: _normalize(getattr(state, key)) for key in golden_state}


def _observation(value):
    return np.asarray(value, dtype=np.float64) if isinstance(value, list) else value


@pytest.mark.parametrize(
    ("golden_name", "algorithm"),
    [
        ("shewhart_control_chart", ShewhartControlChart(learning_period_size=3, window_size=2)),
        ("page_two_sided_cusum", PageTwoSidedCusum(learning_period_size=3, delta=0.1, adaptive_estimation=False)),
        ("crosier_cusum", CrosierCusum(learning_period_size=3, delta=0.1, adaptive_estimation=False)),
        (
            "variance_two_sided_cusum",
            VarianceTwoSidedCUSUM(learning_period_size=4, delta=0.1, adaptive_estimation=False),
        ),
        (
            "autoregressive_cusum",
            AutoregressiveCUSUM(learning_period_size=5, delta=0.1, autoreg_order=1, adaptive_estimation=False),
        ),
        (
            "slope_kl_symbolic_divergence",
            SlopeKLSymbolicDivergence(learning_period_size=3, delta=0.0, gamma=1.0),
        ),
        (
            "univariate_gaussian_conjugate_bocpd",
            UnivariateGaussianConjugateBOCPD(
                learning_period_size=2,
                hazard_lambda=8.0,
                prior_mu=0.0,
                prior_k=1.0,
                prior_alpha=1.0,
                prior_beta=1.0,
                window=5,
                cpf_type=BayesianCPFType.MAX_RUN_LENGTH,
            ),
        ),
    ],
)
def test_concrete_online_algorithm_matches_golden_trace(golden_name: str, algorithm) -> None:
    golden = load_json_golden(Path(__file__).resolve().parents[1] / "golden" / "algorithms" / f"{golden_name}.json")

    values = [float(algorithm.process(_observation(observation))) for observation in golden["observations"]]
    state = algorithm.state

    assert algorithm.name == golden["algorithm"]
    assert np.allclose(values, np.asarray(golden["values"], dtype=np.float64))
    _assert_nested_close(_state_snapshot(state, golden["final_state"]), golden["final_state"])
