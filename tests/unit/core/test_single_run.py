# -*- coding: ascii -*-
"""
Tests for single run.
"""

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.single_run import DetectionTrace, SingleRun, SingleRunDescription
from pysatl_cpd.data.typedefs import TimeseriesAnnotation, frozendict
from pysatl_cpd.typedefs import stable_hash
from tests.support.core import MockDataProvider


def test_single_run_description_hash_uses_stable_hash() -> None:
    detector_description = ChangePointDetectorDescription(name="TestDetector", parameters={"alpha": 1})
    provider_description = TimeseriesAnnotation(
        name="series",
        source="test",
        metadata=frozendict.from_mapping({"fold": 1}),
    )
    description = SingleRunDescription(
        detector_description=detector_description,
        provider_description=provider_description,
    )

    assert hash(description) == stable_hash(
        (
            description.__class__.__module__,
            description.__class__.__qualname__,
            provider_description,
            detector_description,
        )
    )


def test_single_run_description_uses_trace_and_provider_descriptions() -> None:
    detector_description = ChangePointDetectorDescription(name="TestDetector", parameters={"alpha": 1})
    trace = DetectionTrace(detector_description=detector_description)
    provider = MockDataProvider(name="provider-series")

    description = SingleRun(trace=trace, provider=provider).description()

    assert description == SingleRunDescription(
        detector_description=detector_description,
        provider_description=provider.annotation,
    )
