# -*- coding: ascii -*-
"""No-reset benchmark entry point types."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass, field
from typing import Any

from pysatl_cpd.benchmark.online.noreset.detector.noreset_detector import NoResetOnlineDetector
from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import ThresholdsRange
from pysatl_cpd.benchmark.online.noreset.tooling.bisegment_cut import NOOP_BISEGMENT_CUT, BisegmentCut
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithm
from pysatl_cpd.data.providers.transformers.base import IDataTransformer


@dataclass
class OnlineNoResetBenchmarkEntry:
    """A dataclass binding an algorithm, threshold range, and optional data transformer.

    Attributes
    ----------
    algorithm : OnlineAlgorithm
        Algorithm instance to benchmark.
    thresholds : ThresholdsRange
        Range of thresholds to evaluate.
    data_transformer : IDataTransformer or None, optional
        Optional transformer applied to incoming data before processing.
    bisegment_cut : BisegmentCut, optional
        Left/right trim (in samples) applied to bisegment providers before detection.
    """

    algorithm: OnlineAlgorithm[Any, Any, Any]
    thresholds: ThresholdsRange
    data_transformer: IDataTransformer[Any, Any] | None = None
    bisegment_cut: BisegmentCut = field(default_factory=lambda: NOOP_BISEGMENT_CUT)

    @property
    def description(self) -> ChangePointDetectorDescription:
        """Description derived from a throwaway detector instance.

        Includes the entry-level ``bisegment_cut`` so registry key matching
        stays consistent with the detector configuration used in scenarios.

        Returns
        -------
        ChangePointDetectorDescription
        """
        return NoResetOnlineDetector(
            self.algorithm,
            collect_states=False,
            data_transformer=self.data_transformer,
            bisegment_cut=self.bisegment_cut,
        ).description
