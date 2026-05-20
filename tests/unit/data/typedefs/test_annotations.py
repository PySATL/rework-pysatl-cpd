# -*- coding: ascii -*-
"""
Tests for annotations.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.data.typedefs import ProviderType, TimeseriesAnnotation, UnlabeledTimeseriesAnnotation, frozendict
from pysatl_cpd.typedefs import stable_hash


def test_timeseries_annotation_provider_type_is_timeseries() -> None:
    annotation = TimeseriesAnnotation(name="series")

    assert annotation.provider_type is ProviderType.TIMESERIES


def test_unlabeled_timeseries_annotation_provider_type_is_unlabeled() -> None:
    annotation = UnlabeledTimeseriesAnnotation(name="raw")

    assert annotation.provider_type is ProviderType.UNLABELED


def test_unlabeled_timeseries_annotation_hash_uses_stable_hash() -> None:
    annotation = UnlabeledTimeseriesAnnotation(
        name="raw",
        source="sensor",
        metadata=frozendict.from_mapping({"fold": 1}),
    )

    assert hash(annotation) == stable_hash(
        (
            annotation.__class__.__module__,
            annotation.__class__.__qualname__,
            annotation.name,
            annotation.source,
            annotation.metadata,
        )
    )
