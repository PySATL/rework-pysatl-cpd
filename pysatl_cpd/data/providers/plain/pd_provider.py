# -*- coding: ascii -*-
"""Pandas-backed unlabeled data provider."""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from typing import Self

import numpy as np
import pandas as pd

from pysatl_cpd.data.providers.data_provider import DataProvider
from pysatl_cpd.data.typedefs import UnlabeledTimeseriesAnnotation
from pysatl_cpd.typedefs import NumericArray


class PandasDataProvider[AnnotationT: UnlabeledTimeseriesAnnotation](DataProvider[NumericArray, AnnotationT]):
    """Unlabeled pandas-backed provider for numeric feature arrays.

    Parameters
    ----------
    dataset
        Underlying pandas DataFrame.
    annotation
        Annotation for the dataset.
    """

    def __init__(self, dataset: pd.DataFrame, annotation: AnnotationT) -> None:
        self._dataset = dataset.copy().reset_index(drop=True)
        self._annotation = annotation

    def __iter__(self) -> Iterator[NumericArray]:
        """Iterate over rows of the dataset.

        Returns
        -------
        iterator
            Iterator over scalar values for single-column data or row arrays
            for multivariate data.
        """
        if len(self.columns) == 1:
            return iter(self._dataset.to_numpy(dtype=np.float64, copy=False).flatten())
        return iter(self._dataset.to_numpy(dtype=np.float64, copy=False))

    def __len__(self) -> int:
        """Return the number of rows in the dataset.

        Returns
        -------
        length
            Number of samples stored in the provider.
        """
        return len(self._dataset)

    @property
    def annotation(self) -> AnnotationT:
        """
        Get annotation.

        Returns
        -------
        annotation
            The annotation object.
        """
        return self._annotation

    @property
    def dataset(self) -> pd.DataFrame:
        """
        Get dataset copy.

        Returns
        -------
        dataset
            Copy of the underlying DataFrame.
        """
        return self._dataset.copy()

    @property
    def columns(self) -> Sequence[str]:
        """
        Get column names.

        Returns
        -------
        columns
            List of column names.
        """
        return list(self._dataset.columns)

    def select_columns(
        self,
        columns: Sequence[str],
        *,
        rename_provider: bool = False,
    ) -> "PandasDataProvider[AnnotationT]":
        """
        Select subset of columns.

        Parameters
        ----------
        columns
            Column names to select.
        rename_provider
            If True, updates the provider name to reflect the selected columns.

        Returns
        -------
        provider
            New provider with selected columns.

        Raises
        ------
        ValueError
            If no columns are selected or unknown columns are requested.
        """
        if not columns:
            raise ValueError("At least one column must be selected")
        missing = [column for column in columns if column not in self._dataset.columns]
        if missing:
            raise ValueError(f"Unknown columns requested: {missing}")
        annotation = (
            replace(self.annotation, name=f"{self.name}[{','.join(columns)}]") if rename_provider else self.annotation
        )
        return PandasDataProvider(self._dataset[columns], annotation)

    def create_feature_column(
        self,
        *,
        name: str,
        mapping: Callable[[pd.Series], object],
        rename_provider: bool = False,
    ) -> "PandasDataProvider[AnnotationT]":
        """
        Append a derived feature column computed row-wise.

        Parameters
        ----------
        name
            Name of the new feature column.
        mapping
            Callable applied to each row of the dataset.
        rename_provider
            If True, updates the provider name to reflect the new column.

        Returns
        -------
        provider
            New provider with the appended feature column.

        Raises
        ------
        ValueError
            If the column name is empty or already exists.
        """
        if not name:
            raise ValueError("Feature column name must be non-empty")
        if name in self._dataset.columns:
            raise ValueError(f"Feature column '{name}' already exists")

        dataset = self._dataset.copy()
        dataset[name] = dataset.apply(mapping, axis=1)
        annotation = replace(self.annotation, name=f"{self.name}[+{name}]") if rename_provider else self.annotation
        return PandasDataProvider(dataset, annotation)

    def cut(
        self,
        start: int,
        stop: int,
        *,
        annotation: AnnotationT | None = None,
    ) -> "PandasDataProvider[AnnotationT]":
        """
        Slice dataset by row indices.

        Parameters
        ----------
        start
            Start row index (inclusive).
        stop
            Stop row index (inclusive).
        annotation
            Optional annotation to use. If None, generates default.

        Returns
        -------
        provider
            New provider with sliced data.
        """
        self._validate_cut_boundaries(start, stop)
        return PandasDataProvider(
            self._dataset.iloc[start : stop + 1].copy().reset_index(drop=True),
            annotation if annotation is not None else self.default_slice_annotation(start, stop),
        )

    @classmethod
    def merge(
        cls: type[Self],
        providers: Sequence[Self],
        annotation_builder: Callable[[Sequence[AnnotationT]], AnnotationT] | None = None,
    ) -> "PandasDataProvider[AnnotationT]":
        """
        Merge multiple providers.

        Parameters
        ----------
        providers
            Sequence of providers to merge.
        annotation_builder
            Optional function to build merged annotation.

        Returns
        -------
        provider
            New merged provider.

        Raises
        ------
        ValueError
            If providers do not share the same columns and column order.
        """
        cls._validate_merge_inputs(providers)
        first_columns = tuple(providers[0]._dataset.columns)
        for provider in providers[1:]:
            if tuple(provider._dataset.columns) != first_columns:
                raise ValueError("All providers must share the same columns and column order")
        if annotation_builder is None:
            annotation_builder = cls.default_merge_annotation_builder()

        merged_dataset = pd.concat([p._dataset for p in providers], axis=0, ignore_index=True)
        merged_annotation = annotation_builder([p.annotation for p in providers])
        return PandasDataProvider[AnnotationT](merged_dataset, merged_annotation)
