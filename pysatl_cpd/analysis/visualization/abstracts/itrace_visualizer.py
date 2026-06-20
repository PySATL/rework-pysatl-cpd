# -*- coding: ascii -*-
"""
Trace visualizer interface.

This module defines the abstract source class for visualizers that render
detection traces, including detection function values and algorithm metadata.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod
from typing import Self

from pysatl_cpd.analysis.visualization.abstracts.ivisualizer import IVisualizer
from pysatl_cpd.core.detection_trace import DetectionTrace


class ITraceVisualizer[DetectionTraceT: DetectionTrace](IVisualizer, ABC):
    """
    Abstract source class for trace visualizers.

    Visualizers of this type render detection results, including detection
    function values and algorithm metadata.

    Notes
    -----
    The generic DetectionTraceT type is bound by DetectionTrace. This allows
    visualizers to work with specific trace implementations such as
    OnlineDetectionTrace or OfflineDetectionTrace and ensures that any
    concrete implementation works with valid detection traces.
    """

    @abstractmethod
    def set_trace(self, trace: DetectionTraceT) -> Self:  # pragma: no cover
        """
        Set the detection trace to visualize.

        Parameters
        ----------
        trace
            Detection results containing change-point indices, scores,
            and algorithm-specific metadata.

        Returns
        -------
        Self
            Returns self to allow method chaining.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError
