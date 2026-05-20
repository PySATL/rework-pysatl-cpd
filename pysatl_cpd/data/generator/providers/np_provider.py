# -*- coding: ascii -*-
"""Plain provider builders for generated data."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.data.generator.models import GeneratedSeries
from pysatl_cpd.data.providers.labeled import PlainMultivariateLabeledData, PlainUnivariateLabeledData
from pysatl_cpd.data.providers.plain.np_multivariate import NDArrayMultivariateProvider
from pysatl_cpd.data.providers.plain.np_univariate import NDArrayUnivariateProvider
from pysatl_cpd.data.typedefs import TimeseriesAnnotation, UnlabeledTimeseriesAnnotation


def build_plain_multivariate_labeled_data(
    series: GeneratedSeries,
    *,
    name: str,
    annotation: TimeseriesAnnotation | None = None,
) -> PlainMultivariateLabeledData[TimeseriesAnnotation]:
    """
    Build plain multivariate labeled data from generated series.

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
        Plain multivariate labeled data instance.
    """
    effective_annotation = annotation or TimeseriesAnnotation(name=name, metadata=series.metadata)
    return PlainMultivariateLabeledData(
        NDArrayMultivariateProvider(
            series.data,
            UnlabeledTimeseriesAnnotation(
                name=name,
                source=effective_annotation.source,
                metadata=effective_annotation.metadata,
            ),
        ),
        list(series.segments),
        annotation=effective_annotation,
    )


def build_plain_univariate_labeled_data(
    series: GeneratedSeries,
    *,
    feature_name: str,
    name: str,
    annotation: TimeseriesAnnotation | None = None,
) -> PlainUnivariateLabeledData[TimeseriesAnnotation]:
    """
    Build plain univariate labeled data from generated series.

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
        Plain univariate labeled data instance.

    Raises
    ------
    ValueError
        If the feature name is not found in the series.
    """
    if feature_name not in series.feature_names:
        raise ValueError(f"Unknown feature name '{feature_name}'")
    effective_annotation = annotation or TimeseriesAnnotation(name=name, metadata=series.metadata)
    feature_index = series.feature_names.index(feature_name)
    return PlainUnivariateLabeledData(
        NDArrayUnivariateProvider(
            series.data[:, feature_index],
            UnlabeledTimeseriesAnnotation(
                name=name,
                source=effective_annotation.source,
                metadata=effective_annotation.metadata,
            ),
        ),
        list(series.segments),
        annotation=effective_annotation,
    )
