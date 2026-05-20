# -*- coding: ascii -*-
"""No-change policy for no-reset ARL evaluation."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import numpy as np

from pysatl_cpd.benchmark.online.noreset.detector.noreset_trace import NoResetDetectionTrace
from pysatl_cpd.benchmark.online.noreset.metrics.policy.base import (
    _build_noreset_run,
    _point_mask,
    _validate_no_change_run,
)
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.providers.labeled import LabeledData


class NoChangePolicy[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]]:
    """Policy that marks only the first qualifying detection for ARL evaluation.

    Parameters
    ----------
    strict
        Whether to use strict inequality when comparing detection
        function values against the threshold (default True).
    """

    def __init__(self, *, strict: bool = True) -> None:
        self._strict = strict

    def apply(
        self,
        run: SingleRun[OnlineDetectionTrace[StateT], ProviderT],
        threshold: float,
    ) -> SingleRun[NoResetDetectionTrace[StateT], ProviderT]:
        """Apply the policy and keep only the first qualifying detection.

        Validates the run as a no-change run, finds all points where
        the detection function exceeds the threshold, and retains at
        most the first such point.

        Parameters
        ----------
        run
            Input run with an ``OnlineDetectionTrace`` and labeled data.
        threshold
            Threshold applied to the detection function values.

        Returns
        -------
        SingleRun[NoResetDetectionTrace, ProviderT]
            Run wrapping a no-reset trace with at most one detection point.
        """
        _validate_no_change_run(run)
        points = np.flatnonzero(_point_mask(run.trace.detection_function, threshold, self._strict)).tolist()
        return _build_noreset_run(run, threshold, points[:1])
