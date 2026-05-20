# -*- coding: ascii -*-
"""
Single-run analysis: one detection trace on one labeled data provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData
from pysatl_cpd.data.typedefs.annotations import TimeseriesAnnotation
from pysatl_cpd.typedefs import stable_hash

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

__all__ = [
    "SingleRun",
    "SingleRunDescription",
]

AnnotationT = TypeVar("AnnotationT", bound=TimeseriesAnnotation, covariant=True)
ProviderT = TypeVar("ProviderT", bound=LabeledData[Any, Any], covariant=True)
TraceT = TypeVar("TraceT", bound=DetectionTrace, covariant=True)


@dataclass(frozen=True, kw_only=True)
class SingleRunDescription(Generic[AnnotationT]):  # noqa: UP046
    """Hashable description of a single run.

    Combines the detector and data provider metadata so that runs can
    be compared, cached, or used as dictionary keys.

    Attributes
    ----------
    detector_description
        Description of the detector used for the run.
    provider_description
        Annotation of the data provider used for the run.
    """

    detector_description: ChangePointDetectorDescription
    provider_description: AnnotationT

    def __hash__(self) -> int:
        """Hash based on provider and detector description."""
        return stable_hash(
            (
                type(self).__module__,
                type(self).__qualname__,
                self.provider_description,
                self.detector_description,
            )
        )


@dataclass(frozen=True, kw_only=True)
class SingleRun(Generic[TraceT, ProviderT]):  # noqa: UP046
    """
    One run: a trace and the labeled series it was produced from.

    ``P`` is :class:`LabeledData` so that :meth:`description` can read
    :attr:`~LabeledData.annotation` (e.g. ``PandasLabeledData``).
    """

    trace: TraceT
    provider: ProviderT

    def description(self) -> SingleRunDescription[Any]:
        """Build a hashable description of this run.

        Returns
        -------
        SingleRunDescription
            Description combining the detector and provider metadata.
        """
        detector_description = self.trace.detector_description
        return SingleRunDescription(
            provider_description=self.provider.annotation,
            detector_description=detector_description,
        )
