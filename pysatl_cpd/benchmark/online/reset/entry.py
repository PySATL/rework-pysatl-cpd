# -*- coding: ascii -*-
"""Reset online benchmark entry point types."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass

from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.online import OnlineResetDetector


@dataclass
class OnlineResetBenchmarkEntry:
    """A dataclass wrapping an ``OnlineResetDetector`` for benchmark use.

    Attributes
    ----------
    detector
        The reset online detector to benchmark.
    """

    detector: OnlineResetDetector

    @property
    def description(self) -> ChangePointDetectorDescription:
        """Description extracted from the wrapped detector.

        Returns
        -------
        ChangePointDetectorDescription
        """
        return self.detector.description
