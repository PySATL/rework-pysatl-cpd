# -*- coding: ascii -*-
"""
Tests for data transformer contract applications.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.data.providers.transformers import ColumnFeatureCreator, ColumnsSelectorTransformer
from pysatl_cpd.data.providers.transformers.base import ComposedTransformer
from tests.contracts.data.test_data_transformer_contract import DataTransformerContract


class TestColumnsSelectorTransformerContract(DataTransformerContract):
    @pytest.fixture
    def transformer(self) -> ColumnsSelectorTransformer:
        return ColumnsSelectorTransformer(["x"], rename_provider=True)

    def test_transform_returns_expected_type(self, transformer: ColumnsSelectorTransformer, provider) -> None:
        result = transformer.transform(provider)
        assert type(result) is type(provider)
        assert result.feature_columns == ["x"]


class TestColumnFeatureCreatorContract(DataTransformerContract):
    @pytest.fixture
    def transformer(self) -> ColumnFeatureCreator:
        return ColumnFeatureCreator(name="sum", mapping=lambda row: row["x"] + row["y"], rename_provider=True)

    def test_transform_returns_expected_type(self, transformer: ColumnFeatureCreator, provider) -> None:
        result = transformer.transform(provider)
        assert type(result) is type(provider)
        assert "sum" in result.feature_columns


def test_composed_transformer_requires_at_least_one_transformer() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ComposedTransformer()


def test_composed_transformer_annotation_chains_components() -> None:
    feature_creator = ColumnFeatureCreator(name="sum", mapping=lambda row: row["x"] + row["y"])
    transformer = feature_creator & ColumnsSelectorTransformer(["sum"])
    assert transformer.annotation == "feature[sum]->column[sum]"
