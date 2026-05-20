# -*- coding: ascii -*-
"""No-reset detection trace wrapper."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import cast

import numpy as np

from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState
from pysatl_cpd.typedefs import UnivariateNumericArray


class NoResetDetectionTrace[StateT: OnlineAlgorithmState](OnlineDetectionTrace[StateT]):
    """Trace derived from an infinite-threshold run for no-reset benchmarking."""

    @classmethod
    def from_inf_trace(
        cls,
        source_trace: OnlineDetectionTrace[StateT],
        detected_change_points: list[int],
        threshold: float,
    ) -> NoResetDetectionTrace[StateT]:
        """Build a no-reset trace from an infinite-threshold source trace.

        Copies detection function values and algorithm states from the
        source while injecting the detected change points and threshold
        determined by a policy. Processing time is left empty.

        Parameters
        ----------
        source_trace
            Trace produced by an infinite-threshold run.
        detected_change_points
            Change-point indices selected by the policy.
        threshold
            Threshold value that was applied during policy evaluation.

        Returns
        -------
        NoResetDetectionTrace
            A trace ready for classification or ARL analysis.
        """
        empty_processing_time: UnivariateNumericArray = cast(
            UnivariateNumericArray,
            np.array([], dtype=np.float64),
        )
        return cls(
            detector_description=source_trace.detector_description,
            detected_change_points=detected_change_points,
            threshold=threshold,
            detection_function=source_trace.detection_function,
            processing_time=empty_processing_time,
            algorithm_states=source_trace.algorithm_states,
            skip_periods=source_trace.skip_periods,
            learning_periods=source_trace.learning_periods,
            forced_change_points=[],
            signal_change_points=[],
        )
