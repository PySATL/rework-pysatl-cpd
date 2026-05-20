# -*- coding: ascii -*-
"""
Tests for providers.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from pysatl_cpd.data import (
    NDArrayMultivariateProvider,
    NDArrayUnivariateProvider,
    PandasLabeledData,
    PlainMultivariateLabeledData,
    PlainUnivariateLabeledData,
)
from pysatl_cpd.data.providers.plain.pd_provider import PandasDataProvider
from pysatl_cpd.data.typedefs import (
    NoChangeSeriesAnnotation,
    SegmentInfo,
    StateDescriptor,
    TimeseriesAnnotation,
    TransitionDescriptor,
    UnlabeledTimeseriesAnnotation,
)
from pysatl_cpd.typedefs import frozendict


def make_state(label: str, **kwargs: str | int | float | bool) -> StateDescriptor:
    """Build a state descriptor with a stable label field."""
    return StateDescriptor(label=label, **kwargs)


def make_transition(curr_label: str, next_label: str) -> TransitionDescriptor:
    """Build a transition descriptor from two labels."""
    return TransitionDescriptor(curr_state=make_state(curr_label), next_state=make_state(next_label))


def make_segments(
    *,
    lengths: Sequence[int] = (3, 3),
    labels: Sequence[str] = ("baseline", "shift"),
) -> list[SegmentInfo]:
    """Build contiguous inclusive segment descriptors."""
    start = 0
    segments: list[SegmentInfo] = []
    for idx, (length, label) in enumerate(zip(lengths, labels, strict=True)):
        stop = start + length - 1
        segments.append(
            SegmentInfo(
                segment_num=idx,
                segment_start=start,
                segment_end=stop,
                state=make_state(label),
            )
        )
        start = stop + 1
    return segments


def _timeseries_annotation(name: str, source: str = "tests") -> TimeseriesAnnotation:
    return TimeseriesAnnotation(name=name, source=source, metadata=frozendict.from_mapping({"fixture": name}))


def _unlabeled_annotation(name: str, source: str = "tests") -> UnlabeledTimeseriesAnnotation:
    return UnlabeledTimeseriesAnnotation(name=name, source=source, metadata=frozendict.from_mapping({"fixture": name}))


def make_univariate_provider(
    data: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    *,
    name: str = "series",
) -> NDArrayUnivariateProvider[UnlabeledTimeseriesAnnotation]:
    """Build a deterministic plain univariate provider."""
    return NDArrayUnivariateProvider(np.asarray(data, dtype=np.float64), _unlabeled_annotation(name))


def make_multivariate_provider(
    data: Sequence[Sequence[float]] = ((1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0), (5.0, 50.0), (6.0, 60.0)),
    *,
    name: str = "series",
) -> NDArrayMultivariateProvider[UnlabeledTimeseriesAnnotation]:
    """Build a deterministic plain multivariate provider."""
    return NDArrayMultivariateProvider(np.asarray(data, dtype=np.float64), _unlabeled_annotation(name))


def make_pandas_provider(
    frame: pd.DataFrame | None = None,
    *,
    name: str = "series",
) -> PandasDataProvider[UnlabeledTimeseriesAnnotation]:
    """Build a deterministic pandas provider."""
    dataset = (
        frame
        if frame is not None
        else pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
    )
    return PandasDataProvider(dataset, _unlabeled_annotation(name))


def make_univariate_labeled(
    data: Sequence[float] = (1.0, 2.0, 3.0, 10.0, 11.0, 12.0),
    *,
    name: str = "series",
    segments: Iterable[SegmentInfo] | None = None,
) -> PlainUnivariateLabeledData[TimeseriesAnnotation]:
    """Build a labeled univariate provider."""
    provider = make_univariate_provider(data, name=name)
    labeling = list(segments) if segments is not None else make_segments()
    return PlainUnivariateLabeledData.from_unlabeled_data(provider, labeling, _timeseries_annotation(name))


def make_multivariate_labeled(
    data: Sequence[Sequence[float]] = (
        (1.0, 10.0),
        (2.0, 20.0),
        (3.0, 30.0),
        (10.0, 100.0),
        (11.0, 110.0),
        (12.0, 120.0),
    ),
    *,
    name: str = "series",
    segments: Iterable[SegmentInfo] | None = None,
) -> PlainMultivariateLabeledData[TimeseriesAnnotation]:
    """Build a labeled multivariate provider."""
    provider = make_multivariate_provider(data, name=name)
    labeling = list(segments) if segments is not None else make_segments()
    return PlainMultivariateLabeledData.from_unlabeled_data(provider, labeling, _timeseries_annotation(name))


def make_pandas_labeled(
    frame: pd.DataFrame | None = None,
    *,
    name: str = "series",
    segments: Iterable[SegmentInfo] | None = None,
) -> PandasLabeledData[TimeseriesAnnotation]:
    """Build a labeled pandas provider."""
    provider = make_pandas_provider(frame, name=name)
    labeling = list(segments) if segments is not None else make_segments()
    return PandasLabeledData.from_unlabeled_data(provider, labeling, _timeseries_annotation(name))


def make_no_change_labeled(
    data: Sequence[float] = (1.0, 2.0, 3.0, 4.0),
    *,
    name: str = "steady",
    state_label: str = "baseline",
) -> PlainUnivariateLabeledData[NoChangeSeriesAnnotation]:
    """Build a one-segment no-change labeled provider."""
    provider = make_univariate_provider(data, name=name)
    state = make_state(state_label)
    annotation = NoChangeSeriesAnnotation(name=name, source="tests", state=state)
    segments = [SegmentInfo(segment_num=0, segment_start=0, segment_end=len(data) - 1, state=state)]
    return PlainUnivariateLabeledData.from_unlabeled_data(provider, segments, annotation)
