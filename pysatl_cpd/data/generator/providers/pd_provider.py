# -*- coding: ascii -*-
"""Pandas provider builders for generated data."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pandas as pd

from pysatl_cpd.data.generator.models import GeneratedSeries
from pysatl_cpd.data.providers.labeled import PandasLabeledData
from pysatl_cpd.data.providers.labeled.segments_labeling import SegmentsLabeling
from pysatl_cpd.data.providers.plain.pd_provider import PandasDataProvider as PandasUnlabeledDataProvider
from pysatl_cpd.data.typedefs import TimeseriesAnnotation, UnlabeledTimeseriesAnnotation


def build_pandas_labeled_data(
    series: GeneratedSeries,
    *,
    name: str,
    annotation: TimeseriesAnnotation | None = None,
) -> PandasLabeledData[TimeseriesAnnotation]:
    """
    Build pandas labeled data from generated series.

    Parameters
    ----------
    series
        Generated series containing data and metadata.
    name
        Name for the timeseries annotation.
    annotation
        Optional timeseries annotation. If None, one is created
        using the series metadata.

    Returns
    -------
    labeled_data
        Pandas labeled data instance.
    """
    effective_annotation = annotation or TimeseriesAnnotation(name=name, metadata=series.metadata)
    unlabeled = PandasUnlabeledDataProvider(
        pd.DataFrame(series.data, columns=series.feature_names),
        UnlabeledTimeseriesAnnotation(
            name=name,
            source=effective_annotation.source,
            metadata=effective_annotation.metadata,
        ),
    )
    return PandasLabeledData(unlabeled, SegmentsLabeling(series.segments), effective_annotation)


def build_pandas_univariate_labeled_data(
    series: GeneratedSeries,
    *,
    feature_name: str,
    name: str,
    annotation: TimeseriesAnnotation | None = None,
) -> PandasLabeledData[TimeseriesAnnotation]:
    """
    Build pandas univariate labeled data from generated series.

    Parameters
    ----------
    series
        Generated series containing data and metadata.
    feature_name
        Name of the feature to extract as univariate data.
    name
        Name for the timeseries annotation.
    annotation
        Optional timeseries annotation. If None, one is created
        using the series metadata.

    Returns
    -------
    labeled_data
        Pandas univariate labeled data instance.

    Raises
    ------
    ValueError
        If the feature name is not found in the series.
    """
    if feature_name not in series.feature_names:
        raise ValueError(f"Unknown feature name '{feature_name}'")
    effective_annotation = annotation or TimeseriesAnnotation(name=name, metadata=series.metadata)
    feature_index = series.feature_names.index(feature_name)
    unlabeled = PandasUnlabeledDataProvider(
        pd.DataFrame({feature_name: series.data[:, feature_index]}),
        UnlabeledTimeseriesAnnotation(
            name=name,
            source=effective_annotation.source,
            metadata=effective_annotation.metadata,
        ),
    )
    return PandasLabeledData(unlabeled, SegmentsLabeling(series.segments), effective_annotation)
