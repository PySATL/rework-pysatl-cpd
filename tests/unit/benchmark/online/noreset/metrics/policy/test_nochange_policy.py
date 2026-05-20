# -*- coding: ascii -*-
"""
Tests for nochange policy.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from types import SimpleNamespace

import numpy as np

from pysatl_cpd.benchmark.online.noreset.metrics.policy.nochange import NoChangePolicy
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.typedefs import ProviderType


class TestNoChangePolicy:
    def test_init_strict_true(self) -> None:
        policy = NoChangePolicy(strict=True)
        assert policy._strict is True

    def test_init_strict_false(self) -> None:
        policy = NoChangePolicy(strict=False)
        assert policy._strict is False

    def _make_run(self, detection_function: np.ndarray) -> SingleRun:
        from pysatl_cpd.benchmark.online.noreset.detector.noreset_trace import NoResetDetectionTrace
        from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription

        trace = NoResetDetectionTrace(
            detector_description=ChangePointDetectorDescription(name="test"),
            detection_function=detection_function,
            processing_time=np.array([], dtype=np.float64),
            algorithm_states=[],
        )
        provider = SimpleNamespace(
            annotation=SimpleNamespace(provider_type=ProviderType.NO_CHANGE),
        )
        return SingleRun(trace=trace, provider=provider)

    def test_apply_no_detections_above_threshold(self) -> None:
        policy = NoChangePolicy(strict=True)
        run = self._make_run(np.array([0.1, 0.2, 0.3]))

        result = policy.apply(run, threshold=1.0)
        assert len(result.trace.detected_change_points) == 0

    def test_apply_first_detection_above_threshold(self) -> None:
        policy = NoChangePolicy(strict=True)
        run = self._make_run(np.array([0.1, 0.5, 1.5, 2.0]))

        result = policy.apply(run, threshold=1.0)
        assert len(result.trace.detected_change_points) == 1
        assert result.trace.detected_change_points[0] == 2

    def test_apply_multiple_detections_keeps_only_first(self) -> None:
        policy = NoChangePolicy(strict=True)
        run = self._make_run(np.array([0.1, 1.5, 2.0, 3.0]))

        result = policy.apply(run, threshold=1.0)
        assert len(result.trace.detected_change_points) == 1
        assert result.trace.detected_change_points[0] == 1

    def test_apply_strict_false_boundary(self) -> None:
        policy = NoChangePolicy(strict=False)
        run = self._make_run(np.array([0.5, 1.0, 1.5]))

        result = policy.apply(run, threshold=1.0)
        assert len(result.trace.detected_change_points) == 1
        assert result.trace.detected_change_points[0] == 1
