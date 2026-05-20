# -*- coding: ascii -*-
"""Column-oriented transformers for pandas providers."""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any, TypeVar

import pandas as pd

from pysatl_cpd.data.providers.labeled import PandasLabeledData
from pysatl_cpd.data.providers.plain.pd_provider import PandasDataProvider
from pysatl_cpd.data.providers.transformers.base import IDataTransformer

type ColumnSelector = str | Sequence[str]

PandasProviderT = TypeVar("PandasProviderT", PandasDataProvider[Any], PandasLabeledData[Any])


class ColumnsSelectorTransformer(IDataTransformer[PandasProviderT, PandasProviderT]):
    """
    Transformer that selects specific columns from labeled data.

    Parameters
    ----------
    columns
        Column name or sequence of column names to select.
    rename_provider
        Whether to update the provider annotation name.
    """

    def __init__(self, columns: ColumnSelector, *, rename_provider: bool = False) -> None:
        self._columns = columns
        self._rename_provider = rename_provider

    @property
    def _selected_columns(self) -> list[str]:
        """
        Get list of selected column names.

        Returns
        -------
        columns
            List of column names to select.

        Raises
        ------
        ValueError
            If the column list is empty or contains non-string entries.
        """
        if isinstance(self._columns, str):
            return [self._columns]

        columns = list(self._columns)
        if not columns:
            raise ValueError("At least one column must be selected")
        if not all(isinstance(column, str) for column in columns):
            raise ValueError("ColumnsSelectorTransformer only supports column names")
        return columns

    @property
    def annotation(self) -> str:
        """
        Transformer annotation string.

        Returns
        -------
        annotation
            Annotation describing selected columns.
        """
        selected = self._selected_columns
        return f"column[{selected[0]}]" if len(selected) == 1 else "column[" + ";".join(selected) + "]"

    def transform(self, provider: PandasProviderT) -> PandasProviderT:
        """
        Apply column selection to provider.

        Parameters
        ----------
        provider
            Input provider to transform.

        Returns
        -------
        result
            Provider with selected columns.

        Raises
        ------
        TypeError
            If provider is not a PandasLabeledData.
        """
        if not isinstance(provider, PandasLabeledData):
            raise TypeError("ColumnsSelectorTransformer only supports PandasLabeledData providers")
        return provider.select_columns(feature_columns=self._selected_columns, rename_provider=self._rename_provider)


class ColumnFeatureCreator(IDataTransformer[PandasProviderT, PandasProviderT]):
    """Transformer that appends a derived feature column.

    Parameters
    ----------
    name
        Name of the derived feature column.
    mapping
        Callable applied to each pandas row.
    rename_provider
        Whether to update the provider annotation name.
    """

    def __init__(self, name: str, mapping: Any, *, rename_provider: bool = False) -> None:
        self._name = name
        self._mapping = mapping
        self._rename_provider = rename_provider

    @property
    def annotation(self) -> str:
        """Transformer annotation string."""
        return f"feature[{self._name}]"

    def transform(self, provider: PandasProviderT) -> PandasProviderT:
        """
        Append a derived feature column to a pandas provider.

        Parameters
        ----------
        provider
            Input pandas provider to transform.

        Returns
        -------
        result
            Provider with the appended feature column.

        Raises
        ------
        TypeError
            If provider is not a PandasDataProvider or PandasLabeledData,
            or if mapping is not callable.
        ValueError
            If the feature column name is empty.
        """
        if not isinstance(provider, PandasDataProvider | PandasLabeledData):
            raise TypeError("ColumnFeatureCreator only supports PandasDataProvider and PandasLabeledData providers")
        if not self._name:
            raise ValueError("Feature column name must be non-empty")
        if not callable(self._mapping):
            raise TypeError("ColumnFeatureCreator mapping must be callable")
        return provider.create_feature_column(
            name=self._name,
            mapping=self._row_mapping,
            rename_provider=self._rename_provider,
        )

    def _row_mapping(self, row: pd.Series) -> object:
        """Apply the user-supplied mapping to a single row."""
        return self._mapping(row)
