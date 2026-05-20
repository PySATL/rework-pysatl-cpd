# -*- coding: ascii -*-
"""Online change-point detector interface."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from typing import Any, Self

from pysatl_cpd.core.change_point_detector import ChangePointDetector
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.core.online.ionline_algorithm import (
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
)
from pysatl_cpd.data import DataProvider
from pysatl_cpd.data.providers.transformers.base import IDataTransformer
from pysatl_cpd.data.typedefs import TimeseriesAnnotation


class OnlineDetector[
    DataT,
    ConfigurationT: OnlineAlgorithmConfiguration,
    StateT: OnlineAlgorithmState,
](ChangePointDetector[DataT], ABC):
    """Online detector that owns an online algorithm and returns an online trace.

    Parameters
    ----------
    algorithm
        Algorithm instance that drives the per-step detection.
    data_transformer
        Optional transformer applied to incoming data before processing.

    Attributes
    ----------
    algorithm
        Algorithm instance that drives the per-step detection.
    """

    algorithm: OnlineAlgorithm[DataT, ConfigurationT, StateT]

    def __init__(
        self,
        algorithm: OnlineAlgorithm[DataT, ConfigurationT, StateT],
        *,
        data_transformer: IDataTransformer[Any, Any] | None = None,
    ) -> None:
        super().__init__(data_transformer=data_transformer)
        self.algorithm = algorithm

    def detect[AnnotationT: TimeseriesAnnotation](
        self, data: DataProvider[DataT, AnnotationT]
    ) -> OnlineDetectionTrace[StateT]:
        """Run the online pipeline over the provider and return a full online trace.

        Parameters
        ----------
        data
            Source of observations to process sequentially.

        Returns
        -------
        OnlineDetectionTrace
            Trace containing per-step detection results and metadata.
        """
        return super().detect(data)  # type: ignore[return-value]

    @abstractmethod
    def clone(self) -> Self:
        """Create an independent detector instance with the same configuration.

        Returns
        -------
        Self
            A new detector instance with identical configuration.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method")  # pragma: no cover

    @abstractmethod
    def _detect[AnnotationT: TimeseriesAnnotation](
        self, data: DataProvider[DataT, AnnotationT]
    ) -> OnlineDetectionTrace[StateT]:
        """Run detection on transformed provider data."""
        raise NotImplementedError("Subclasses must implement this method")  # pragma: no cover
