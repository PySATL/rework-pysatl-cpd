# -*- coding: ascii -*-
"""
Tests for conftest.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from tests.support.algorithms import CountingAlgorithm, CountingAlgorithmConfig, IdentityTransformer
from tests.support.providers import make_state


@pytest.fixture
def base_algorithm() -> CountingAlgorithm:
    """Provide a deterministic online algorithm for contracts."""
    return CountingAlgorithm(CountingAlgorithmConfig())


@pytest.fixture
def identity_transformer() -> IdentityTransformer:
    """Provide a passthrough transformer for detector contracts."""
    return IdentityTransformer()


@pytest.fixture
def baseline_state():
    """Common baseline state descriptor."""
    return make_state("baseline")
