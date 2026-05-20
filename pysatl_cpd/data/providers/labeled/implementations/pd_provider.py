# -*- coding: ascii -*-
"""
Pandas-backed labeled data.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from typing import cast

import pandas as pd

from pysatl_cpd.data.providers.data_provider import DataProvider
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData
from pysatl_cpd.data.providers.labeled.segments_labeling import SegmentsLabeling
from pysatl_cpd.data.providers.plain.pd_provider import PandasDataProvider
from pysatl_cpd.data.typedefs import SegmentInfo, TimeseriesAnnotation, UnlabeledTimeseriesAnnotation
from pysatl_cpd.typedefs import NumericArray


class PandasLabeledData[AnnotationT: TimeseriesAnnotation](LabeledData[NumericArray, AnnotationT]):
    """Pandas labeled data provider with configurable column names.

    Parameters
    ----------
    unlabeled
        Unlabeled data provider for the timeseries.
    segment_info
        Iterable of segment information.
    annotation
        Annotation instance for labeling.
    segment_column
        Name of the segment column.
    segment_start_column
        Name of the segment start column.
    segment_end_column
        Name of the segment end column.
    """

    def __init__(
        self,
        unlabeled: PandasDataProvider[UnlabeledTimeseriesAnnotation],
        segment_info: Iterable[SegmentInfo],
        annotation: AnnotationT,
        *,
        segment_column: str = "segment",
        segment_start_column: str = "start",
        segment_end_column: str = "end",
    ) -> None:
        self._segment_column = segment_column
        self._segment_start_column = segment_start_column
        self._segment_end_column = segment_end_column

        super().__init__(unlabeled, segment_info, annotation)

    @property
    def unlabeled(self) -> PandasDataProvider[UnlabeledTimeseriesAnnotation]:
        """
        Get the underlying unlabeled data provider.

        Returns
        -------
        unlabeled
            The underlying unlabeled data provider.
        """
        return cast(PandasDataProvider[UnlabeledTimeseriesAnnotation], self._unlabeled_data)

    @classmethod
    def from_unlabeled_data[A: TimeseriesAnnotation](
        cls,
        unlabeled: DataProvider[NumericArray, UnlabeledTimeseriesAnnotation],
        segment_info: Iterable[SegmentInfo],
        annotation: A,
    ) -> "PandasLabeledData[A]":
        """
        Create labeled data from unlabeled data provider.

        Parameters
        ----------
        unlabeled
            Unlabeled data provider for the timeseries.
        segment_info
            Iterable of segment information.
        annotation
            Annotation instance for labeling.

        Returns
        -------
        labeled_data
            New labeled data instance.

        Raises
        ------
        TypeError
            If unlabeled is not a PandasDataProvider.
        """
        if not isinstance(unlabeled, PandasDataProvider):
            raise TypeError("PandasLabeledData requires a PandasDataProvider")
        return cast(
            "PandasLabeledData[A]",
            cls(unlabeled, SegmentsLabeling(list(segment_info)), cast(AnnotationT, annotation)),
        )

    def dataset(self, state_columns: dict[str, str] | None = None) -> pd.DataFrame:
        """
        Get the dataset with segment columns.

        Parameters
        ----------
        state_columns
            Optional mapping of column names to state keys.

        Returns
        -------
        dataset
            DataFrame with timeseries and segment columns.
        """
        dataset = self.unlabeled.dataset
        dataset = pd.concat([dataset, self._segment_columns(state_columns)], axis=1)
        return dataset

    @property
    def feature_columns(self) -> Sequence[str]:
        """
        Get the feature column names.

        Returns
        -------
        feature_columns
            Sequence of feature column names.
        """
        return self.unlabeled.columns

    def select_columns(
        self,
        *,
        feature_columns: Sequence[str],
        rename_provider: bool = False,
    ) -> "PandasLabeledData[AnnotationT]":
        """
        Select a subset of feature columns.

        Parameters
        ----------
        feature_columns
            Sequence of column names to select.
        rename_provider
            If True, updates the provider name to reflect the selected columns.

        Returns
        -------
        labeled_data
            New labeled data with selected columns.
        """
        annotation = (
            replace(self.annotation, name=f"{self.name}[{','.join(feature_columns)}]")
            if rename_provider
            else self.annotation
        )
        return PandasLabeledData(
            self.unlabeled.select_columns(feature_columns, rename_provider=rename_provider),
            self.segments_labeling,
            annotation,
        )

    def create_feature_column(
        self,
        *,
        name: str,
        mapping: Callable[[pd.Series], object],
        rename_provider: bool = False,
    ) -> "PandasLabeledData[AnnotationT]":
        """
        Append a derived feature column computed row-wise.

        Parameters
        ----------
        name
            Name of the new feature column.
        mapping
            Callable applied to each feature row.
        rename_provider
            If True, updates the provider name to reflect the new column.

        Returns
        -------
        labeled_data
            New labeled data with the appended feature column.
        """
        annotation = replace(self.annotation, name=f"{self.name}[+{name}]") if rename_provider else self.annotation
        return PandasLabeledData(
            self.unlabeled.create_feature_column(name=name, mapping=mapping, rename_provider=rename_provider),
            self.segments_labeling,
            annotation,
        )

    def _segment_columns(self, state_columns: dict[str, str] | None = None) -> pd.DataFrame:
        """
        Build segment columns DataFrame.

        Parameters
        ----------
        state_columns
            Optional mapping of column names to state keys.

        Returns
        -------
        segment_df
            DataFrame with segment columns.
        """
        df = pd.DataFrame(index=range(len(self)), dtype="object")
        for label in self.segments_labeling:
            df.loc[label.segment_start : label.segment_end, self._segment_column] = label.segment_num
            df.loc[label.segment_start : label.segment_end, self._segment_start_column] = label.segment_start
            df.loc[label.segment_start : label.segment_end, self._segment_end_column] = label.segment_end

        if state_columns is not None:
            for column, key in state_columns.items():
                df[column] = self._make_series_from_state_var(key)

        return df.convert_dtypes()

    def _make_series_from_state_var(self, state_key: str) -> pd.Series:
        """
        Create series from state variable.

        Parameters
        ----------
        state_key
            Key to extract from segment state.

        Returns
        -------
        series
            Series with state values for each segment.
        """
        series = pd.Series(index=range(len(self)), dtype="object")
        for label in self.segments_labeling:
            series.loc[label.segment_start : label.segment_end] = label.state.get(state_key, pd.NA)
        return series.convert_dtypes()
