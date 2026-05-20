# -*- coding: ascii -*-
"""
Tests for bayesian utils.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.algorithms.online import BayesianCPFType
from pysatl_cpd.algorithms.online.bayesian.component.cpf import DropCPF, MaxRunLengthCPF
from pysatl_cpd.algorithms.online.bayesian.component.hazard import ConstantHazard
from pysatl_cpd.algorithms.online.bayesian.component.likelihood import GaussianConjugate
from pysatl_cpd.algorithms.online.bayesian.utils import (
    get_cpf_function,
    get_hazard_function,
    get_likelihood_function,
)


class TestBayesianUtils:
    def test_get_hazard_function_returns_constant_hazard(self) -> None:
        hazard = get_hazard_function(5.0)
        assert isinstance(hazard, ConstantHazard)

    def test_get_likelihood_function_returns_gaussian_conjugate(self) -> None:
        likelihood = get_likelihood_function(0.0, 1.0, 1.0, 1.0)
        assert isinstance(likelihood, GaussianConjugate)

    def test_get_cpf_function_max_run_length(self) -> None:
        cpf = get_cpf_function(BayesianCPFType.MAX_RUN_LENGTH)
        assert isinstance(cpf, MaxRunLengthCPF)

    def test_get_cpf_function_drop(self) -> None:
        cpf = get_cpf_function(BayesianCPFType.DROP)
        assert isinstance(cpf, DropCPF)

    def test_get_cpf_function_invalid_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown cpf_type"):
            get_cpf_function("invalid")  # type: ignore[arg-type]
