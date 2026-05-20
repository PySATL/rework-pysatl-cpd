# -*- coding: ascii -*-
"""
Tests for transformers extra.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pandas as pd

from pysatl_cpd.data.providers.transformers import ColumnFeatureCreator, ColumnsSelectorTransformer
from tests.support.providers import make_pandas_labeled


def test_composed_transformer_can_append_with_and_operator() -> None:
    labeled = make_pandas_labeled(pd.DataFrame({"x": [1.0, 2.0], "y": [10.0, 20.0]}), segments=[])
    transformer = (
        ColumnFeatureCreator(name="sum", mapping=lambda row: row["x"] + row["y"]) & ColumnsSelectorTransformer(["sum"])
    ) & ColumnsSelectorTransformer("sum")

    result = transformer.transform(labeled)

    assert transformer.annotation == "feature[sum]->column[sum]->column[sum]"
    assert result.feature_columns == ["sum"]
    assert list(result.dataset()["sum"]) == [11.0, 22.0]


def test_columns_selector_transformer_accepts_single_string_column() -> None:
    labeled = make_pandas_labeled(pd.DataFrame({"x": [1.0, 2.0], "y": [10.0, 20.0]}), segments=[])

    result = ColumnsSelectorTransformer("x").transform(labeled)

    assert result.feature_columns == ["x"]
