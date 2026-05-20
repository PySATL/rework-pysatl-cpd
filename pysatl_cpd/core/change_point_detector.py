# -*- coding: ascii -*-
"""Offline / batch change-point detector interface."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self, cast

from pysatl_cpd.data import DataProvider
from pysatl_cpd.data.providers.transformers.base import IDataTransformer
from pysatl_cpd.data.typedefs import TimeseriesAnnotation
from pysatl_cpd.typedefs import frozendict, make_hashable, stable_hash

if TYPE_CHECKING:
    from pysatl_cpd.core.detection_trace import DetectionTrace


@dataclass(frozen=True, kw_only=True)
class ChangePointDetectorDescription:
    """Static description of the detector for logging and benchmarking.

    Attributes
    ----------
    name
        Human-readable detector name.
    parameters
        Frozen dictionary of detector parameters. Values are normalised
        to hashable form during ``__post_init__``.
    """

    name: str
    parameters: frozendict[str, Hashable] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        """Normalise parameters to hashable values after initialisation."""
        normalized = frozendict.from_mapping({key: make_hashable(value) for key, value in self.parameters.items()})
        object.__setattr__(self, "parameters", normalized)

    def __hash__(self) -> int:
        """Stable (when parameters are fixed) hash for :class:`DetectionTrace`."""
        return stable_hash((self.name, self.parameters))  # pragma: no cover


# TODO: force implement __copy__()  magic
class ChangePointDetector[DataT](ABC):
    """A detector with a uniform output - :class:`DetectionTrace`.

    Parameters
    ----------
    data_transformer
        Optional transformer applied to incoming data before detection.
    """

    def __init__(self, *, data_transformer: IDataTransformer[Any, Any] | None = None) -> None:
        self.data_transformer = data_transformer

    @property
    @abstractmethod
    def description(self) -> ChangePointDetectorDescription:
        """Return a static description of the detector.

        Returns
        -------
        ChangePointDetectorDescription
            Description containing the detector name and parameters.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this property.
        """
        raise NotImplementedError("Subclasses must implement this method")  # pragma: no cover

    def detect[AnnotationT: TimeseriesAnnotation](self, data: DataProvider[DataT, AnnotationT]) -> DetectionTrace:
        """Apply the detector-level transformer and build a trace.

        Parameters
        ----------
        data
            Source data to run detection on.

        Returns
        -------
        DetectionTrace
            Trace containing the detection results.
        """
        transformed = self._transform_data(data)
        return self._detect(transformed)

    @abstractmethod
    def _detect[AnnotationT: TimeseriesAnnotation](self, data: DataProvider[DataT, AnnotationT]) -> DetectionTrace:
        """Build a trace using transformed provider data.

        Parameters
        ----------
        data
            Transformed provider data ready for detection.

        Returns
        -------
        DetectionTrace
            Trace containing the detection results.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method")  # pragma: no cover

    @abstractmethod
    def clone(self) -> Self:
        """Create an independent detector instance with the same configuration.

        Returns
        -------
        Self
            A new detector instance identical in configuration but with
            no accumulated state.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method")  # pragma: no cover

    def _transform_data[AnnotationT: TimeseriesAnnotation](
        self, data: DataProvider[DataT, AnnotationT]
    ) -> DataProvider[DataT, AnnotationT]:
        """Apply the detector-level data transformer if one is configured."""
        if self.data_transformer is None:
            return data
        transformed = self.data_transformer.transform(data)
        if not isinstance(transformed, DataProvider):
            raise TypeError("data_transformer must return a DataProvider instance")
        return cast(DataProvider[DataT, AnnotationT], transformed)
