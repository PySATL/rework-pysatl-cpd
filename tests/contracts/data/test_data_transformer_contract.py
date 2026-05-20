# -*- coding: ascii -*-
"""
Tests for data transformer contract.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pandas as pd
import pytest

from pysatl_cpd.data.providers import DataProvider
from tests.support.providers import make_pandas_labeled


def _normalized_values(provider: DataProvider) -> list[object]:
    values = list(provider)
    normalized: list[object] = []
    for value in values:
        if isinstance(value, np.ndarray):
            normalized.append(value.tolist())
        elif hasattr(value, "item"):
            normalized.append(value.item())
        else:
            normalized.append(value)
    return normalized


class DataTransformerContract:
    """Reusable checks for transformer implementations."""

    @pytest.fixture
    def transformer(self):
        raise NotImplementedError

    @pytest.fixture
    def provider(self):
        return make_pandas_labeled()

    def test_annotation_is_non_empty_string(self, transformer) -> None:
        assert isinstance(transformer.annotation, str)
        assert transformer.annotation

    def test_transform_returns_data_provider(self, transformer, provider) -> None:
        assert isinstance(transformer.transform(provider), DataProvider)

    def test_transform_is_deterministic(self, transformer, provider) -> None:
        left = transformer.transform(provider)
        right = transformer.transform(provider)
        assert _normalized_values(left) == _normalized_values(right)
        assert left.annotation == right.annotation

    def test_transform_does_not_mutate_source(self, transformer, provider) -> None:
        before = provider.dataset().copy()
        transformer.transform(provider)
        pd.testing.assert_frame_equal(provider.dataset(), before)
