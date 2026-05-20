"""
Tests for metrics policy base.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.benchmark.online.noreset.detector.noreset_trace import NoResetDetectionTrace
from pysatl_cpd.benchmark.online.noreset.metrics.policy.base import (
    _build_noreset_run,
    _event_mask,
    _first_region_point,
    _point_mask,
    _region_points,
    _validate_bisegment_run,
    _validate_no_change_run,
)
from pysatl_cpd.benchmark.online.noreset.metrics.policy.bisegment import (
    BisegmentPolicyBase,
    EventBasedPolicy,
    MixedPolicy,
    PointBasedPolicy,
)
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.typedefs import ProviderType
from pysatl_cpd.typedefs import UnivariateNumericArray
from tests.support.algorithms import CountingAlgorithmState

# ===========================================================================
#  policy/base.py helpers
# ===========================================================================


class TestPointMask:
    def test_strict_greater(self) -> None:
        values = np.array([0.0, 1.0, 2.0, 1.5, 0.5], dtype=np.float64)
        mask = _point_mask(values, 1.0, strict=True)
        expected = np.array([False, False, True, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_non_strict_greater_equal(self) -> None:
        values = np.array([0.0, 1.0, 2.0, 1.5, 0.5], dtype=np.float64)
        mask = _point_mask(values, 1.0, strict=False)
        expected = np.array([False, True, True, True, False])
        np.testing.assert_array_equal(mask, expected)


class TestEventMask:
    def test_empty_array_returns_zeros(self) -> None:
        values = np.array([], dtype=np.float64)
        mask = _event_mask(values, 0.5, strict=True)
        assert mask.shape == (0,)
        assert mask.dtype == np.bool_

    def test_single_element_below(self) -> None:
        values = np.array([0.3], dtype=np.float64)
        mask = _event_mask(values, 0.5, strict=True)
        np.testing.assert_array_equal(mask, [False])

    def test_single_element_above(self) -> None:
        values = np.array([0.7], dtype=np.float64)
        mask = _event_mask(values, 0.5, strict=True)
        np.testing.assert_array_equal(mask, [True])

    def test_first_element_is_event(self) -> None:
        values = np.array([0.8, 0.3, 0.9], dtype=np.float64)
        mask = _event_mask(values, 0.5, strict=True)
        # event mask: prev_mask (first elem always True) & curr_mask (point mask)
        np.testing.assert_array_equal(mask, [True, False, True])

    def test_strict_vs_non_strict(self) -> None:
        values = np.array([0.5, 0.5], dtype=np.float64)
        strict_mask = _event_mask(values, 0.5, strict=True)
        non_strict_mask = _event_mask(values, 0.5, strict=False)
        # strict: 0.5 > 0.5 -> False, so no event
        np.testing.assert_array_equal(strict_mask, [False, False])
        # non-strict: 0.5 >= 0.5 -> True, first elem is event; second follows a non-strict above
        # For 2nd element: curr = 0.5 >= 0.5 -> True, prev = 0.5 <= 0.5 -> True, so True
        np.testing.assert_array_equal(non_strict_mask, [True, True])


class TestValidateNoChangeRun:
    def test_raises_on_non_nochange_provider(self) -> None:
        annotation = _make_annotation(provider_type="bisegment")
        provider = _make_provider(annotation=annotation)
        run = _make_empty_run(provider=provider)
        with pytest.raises(ValueError, match="No-reset no-change policies require no-change providers"):
            _validate_no_change_run(run)

    def test_passes_on_nochange_provider(self) -> None:
        annotation = _make_annotation(provider_type=ProviderType.NO_CHANGE)
        provider = _make_provider(annotation=annotation)
        run = _make_empty_run(provider=provider)
        # Should not raise
        assert _validate_no_change_run(run) is None


class TestValidateBisegmentRun:
    def test_raises_on_non_bisegment_provider_type(self) -> None:
        annotation = _make_annotation(provider_type="segment")
        provider = _make_provider(annotation=annotation, change_points=(5,))
        run = _make_empty_run(provider=provider)
        with pytest.raises(ValueError, match="No-reset classification policies require bisegment providers"):
            _validate_bisegment_run(run)

    def test_raises_on_wrong_change_point_count(self) -> None:
        annotation = _make_annotation(provider_type="bisegment")
        provider = _make_provider(annotation=annotation, change_points=(5, 10))
        run = _make_empty_run(provider=provider)
        with pytest.raises(ValueError, match="No-reset classification policies require exactly one true change point"):
            _validate_bisegment_run(run)

    def test_returns_change_point_for_valid_run(self) -> None:
        annotation = _make_annotation(provider_type="bisegment")
        provider = _make_provider(annotation=annotation, change_points=(42,))
        run = _make_empty_run(provider=provider)
        cp = _validate_bisegment_run(run)
        assert cp == 42


class TestRegionPoints:
    def test_start_greater_than_end_returns_empty(self) -> None:
        mask = np.array([True, False, True], dtype=np.bool_)
        assert _region_points(mask, 3, 1) == []

    def test_returns_marked_indices_in_range(self) -> None:
        mask = np.array([False, True, False, True, True, False], dtype=np.bool_)
        result = _region_points(mask, 1, 4)
        assert result == [1, 3, 4]

    def test_single_element_range(self) -> None:
        mask = np.array([False, True, False], dtype=np.bool_)
        assert _region_points(mask, 2, 2) == []


class TestFirstRegionPoint:
    def test_returns_only_first_point(self) -> None:
        mask = np.array([False, True, False, True, True, False], dtype=np.bool_)
        result = _first_region_point(mask, 1, 4)
        assert result == [1]

    def test_empty_when_no_points(self) -> None:
        mask = np.array([False, False, False], dtype=np.bool_)
        assert _first_region_point(mask, 0, 2) == []


class TestBuildNoresetRun:
    def test_builds_single_run(self) -> None:
        values = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        trace = _make_trace(values)
        provider = _make_provider(
            annotation=_make_annotation(provider_type="bisegment"),
            change_points=(1,),
        )
        run = SingleRun(trace=trace, provider=provider)

        result = _build_noreset_run(run, threshold=1.5, detected_change_points=[2])

        assert isinstance(result, SingleRun)
        assert isinstance(result.trace, NoResetDetectionTrace)
        assert result.trace.threshold == 1.5
        assert result.trace.detected_change_points == [2]
        assert result.provider is provider

    def test_build_noreset_run_deduplicates_and_sorts_points(self) -> None:
        values = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        trace = _make_trace(values)
        provider = _make_provider(
            annotation=_make_annotation(provider_type="bisegment"),
            change_points=(1,),
        )
        run = SingleRun(trace=trace, provider=provider)

        result = _build_noreset_run(run, threshold=0.5, detected_change_points=[2, 0, 2, 1])

        assert result.trace.detected_change_points == [0, 1, 2]


# ===========================================================================
#  policy/bisegment.py  -  BisegmentPolicyBase
# ===========================================================================


class TestBisegmentPolicyBase:
    def test_negative_max_delay_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="max_delay must be non-negative"):
            _BisegmentPolicyConcrete(max_delay=-1)

    def test_zero_max_delay_allowed(self) -> None:
        policy = _BisegmentPolicyConcrete(max_delay=0)
        assert policy._max_delay == 0
        assert policy._strict is True

    def test_non_default_strict(self) -> None:
        policy = _BisegmentPolicyConcrete(max_delay=5, strict=False)
        assert policy._strict is False

    def test_true_region_end_within_bounds(self) -> None:
        policy = _BisegmentPolicyConcrete(max_delay=3)
        # cp=5, length=10 -> end = min(5+3, 9) = 8
        assert policy._true_region_end(5, 10) == 8

    def test_true_region_end_clamped_to_length(self) -> None:
        policy = _BisegmentPolicyConcrete(max_delay=10)
        # cp=5, length=10 -> end = min(5+10, 9) = 9
        assert policy._true_region_end(5, 10) == 9

    def test_true_region_end_zero_length(self) -> None:
        policy = _BisegmentPolicyConcrete(max_delay=0)
        assert policy._true_region_end(0, 1) == 0


# ===========================================================================
#  PointBasedPolicy
# ===========================================================================


class TestPointBasedPolicy:
    def test_select_false_region_point_mask(self) -> None:
        policy = PointBasedPolicy(max_delay=3, strict=True)
        values = np.array([0.1, 0.6, 0.8, 0.3, 0.9, 0.2], dtype=np.float64)
        cp = 4
        # point mask for threshold 0.5: [F, T, T, F, T, F]
        # false region: indices 0..3 -> [1, 2]
        points = policy._select_false_region(values, 0.5, cp)
        assert points == [1, 2]

    def test_select_false_region_no_detections(self) -> None:
        policy = PointBasedPolicy(max_delay=3, strict=True)
        values = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        cp = 4
        points = policy._select_false_region(values, 0.5, cp)
        assert points == []

    def test_select_true_region_first_point(self) -> None:
        policy = PointBasedPolicy(max_delay=3, strict=True)
        values = np.array([0.1, 0.6, 0.8, 0.3, 0.9, 0.2, 0.7], dtype=np.float64)
        cp = 3
        # true region: indices 3..min(3+3,6)=6 -> 3..6
        # point mask at 0.5: [F, T, T, F, T, F, T]
        # first in 3..6 is index 4
        points = policy._select_true_region(values, 0.5, cp)
        assert points == [4]

    def test_select_true_region_no_detection(self) -> None:
        policy = PointBasedPolicy(max_delay=3, strict=True)
        values = np.array([0.1, 0.2, 0.1, 0.3], dtype=np.float64)
        cp = 2
        points = policy._select_true_region(values, 0.5, cp)
        assert points == []

    def test_apply_with_real_run(self) -> None:
        policy = PointBasedPolicy(max_delay=2, strict=True)
        values = np.array([0.1, 0.7, 0.2, 0.8, 0.9, 0.1], dtype=np.float64)
        run = _make_bisegment_run(values, change_point=2)

        result = policy.apply(run, 0.5)

        assert isinstance(result, SingleRun)
        assert isinstance(result.trace, NoResetDetectionTrace)
        # false region: indices 0..1, point mask [F,T] -> [1]
        # true region: first point in 2..min(2+2,5)=2..4, mask [F,T,T] -> first at 3
        assert result.trace.detected_change_points == [1, 3]


# ===========================================================================
#  EventBasedPolicy
# ===========================================================================


class TestEventBasedPolicy:
    def test_select_false_region_event_mask(self) -> None:
        policy = EventBasedPolicy(max_delay=3, strict=True)
        values = np.array([0.6, 0.3, 0.8, 0.7, 0.2, 0.9], dtype=np.float64)
        cp = 4
        # event mask at 0.5: [T, F, T, F, F, T]  (first above, followed by below, then crossing)
        # false region: indices 0..3 -> [0, 2]
        points = policy._select_false_region(values, 0.5, cp)
        assert points == [0, 2]

    def test_select_true_region_first_event(self) -> None:
        policy = EventBasedPolicy(max_delay=3, strict=True)
        values = np.array([0.1, 0.6, 0.2, 0.7, 0.8, 0.3], dtype=np.float64)
        cp = 2
        # event mask at 0.5: [F, T, F, T, F, F]
        # true region: 2..min(2+3,5) = 2..5 -> events at 3
        points = policy._select_true_region(values, 0.5, cp)
        assert points == [3]

    def test_apply_with_real_run(self) -> None:
        policy = EventBasedPolicy(max_delay=2, strict=True)
        values = np.array([0.6, 0.3, 0.7, 0.2, 0.9, 0.1], dtype=np.float64)
        run = _make_bisegment_run(values, change_point=2)

        result = policy.apply(run, 0.5)

        assert isinstance(result, SingleRun)
        assert isinstance(result.trace, NoResetDetectionTrace)
        # false region: indices 0..1, event mask [T, F] -> [0]
        # true region: first in 2..4, event mask [T, F, T] -> first at 2
        assert result.trace.detected_change_points == [0, 2]


# ===========================================================================
#  MixedPolicy
# ===========================================================================


class TestMixedPolicy:
    def test_select_false_region_uses_event_mask(self) -> None:
        policy = MixedPolicy(max_delay=3, strict=True)
        values = np.array([0.6, 0.3, 0.8, 0.7, 0.2, 0.9], dtype=np.float64)
        cp = 4
        points = policy._select_false_region(values, 0.5, cp)
        # event mask: [T, F, T, F, F, T], false region 0..3 -> [0, 2]
        assert points == [0, 2]

    def test_select_true_region_uses_point_mask(self) -> None:
        policy = MixedPolicy(max_delay=3, strict=True)
        values = np.array([0.1, 0.6, 0.2, 0.7, 0.8, 0.3], dtype=np.float64)
        cp = 2
        # point mask at 0.5: [F, T, F, T, T, F]
        # true region: 2..5 -> first point at 3
        points = policy._select_true_region(values, 0.5, cp)
        assert points == [3]

    def test_apply_with_real_run(self) -> None:
        policy = MixedPolicy(max_delay=2, strict=True)
        values = np.array([0.6, 0.3, 0.7, 0.8, 0.2, 0.1], dtype=np.float64)
        run = _make_bisegment_run(values, change_point=2)

        result = policy.apply(run, 0.5)

        assert isinstance(result, SingleRun)
        assert isinstance(result.trace, NoResetDetectionTrace)
        # false region (event): 0..1, event mask [T, F] -> [0]
        # true region (point, strict): true region 2..4, point mask [T, T, F] -> first at 2
        assert result.trace.detected_change_points == [0, 2]


# ===========================================================================
# All policies - apply validation (error paths)
# ===========================================================================


class TestBisegmentPoliciesApplyValidation:
    @pytest.mark.parametrize("policy_cls", [PointBasedPolicy, EventBasedPolicy, MixedPolicy])
    def test_apply_raises_on_non_bisegment(self, policy_cls) -> None:
        policy = policy_cls(max_delay=3)
        annotation = _make_annotation(provider_type="segment")
        provider = _make_provider(annotation=annotation, change_points=(1,))
        trace = _make_trace(np.array([0.1, 0.2], dtype=np.float64))
        run = SingleRun(trace=trace, provider=provider)
        with pytest.raises(ValueError, match="No-reset classification policies require bisegment providers"):
            policy.apply(run, 0.5)

    @pytest.mark.parametrize("policy_cls", [PointBasedPolicy, EventBasedPolicy, MixedPolicy])
    def test_apply_raises_on_wrong_cp_count(self, policy_cls) -> None:
        policy = policy_cls(max_delay=3)
        annotation = _make_annotation(provider_type="bisegment")
        provider = _make_provider(annotation=annotation, change_points=(1, 5))
        trace = _make_trace(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64))
        run = SingleRun(trace=trace, provider=provider)
        with pytest.raises(ValueError, match="No-reset classification policies require exactly one true change point"):
            policy.apply(run, 0.5)


# ===========================================================================
#  Helpers
# ===========================================================================


class _Annotation:
    """Minimal annotation that just holds a provider_type value."""

    def __init__(self, provider_type: str | ProviderType) -> None:
        self.provider_type = provider_type


class _Provider:
    """Minimal provider with annotation and change_points."""

    def __init__(self, annotation: object, change_points: tuple[int, ...] = ()) -> None:
        self.annotation = annotation
        self.change_points = change_points


class _BisegmentPolicyConcrete(BisegmentPolicyBase):
    """Concrete subclass to allow testing base-class methods."""

    def _select_false_region(self, values, threshold, cp):
        return []

    def _select_true_region(self, values, threshold, cp):
        return []


def _make_annotation(provider_type: str | ProviderType) -> _Annotation:
    return _Annotation(provider_type=provider_type)


def _make_provider(annotation: object, change_points: tuple[int, ...] = ()) -> _Provider:
    return _Provider(annotation=annotation, change_points=change_points)


def _make_empty_run(provider: object) -> SingleRun:
    trace = _make_trace(np.array([], dtype=np.float64))
    return SingleRun(trace=trace, provider=provider)  # type: ignore[arg-type]


def _make_trace(values: UnivariateNumericArray) -> OnlineDetectionTrace:
    return OnlineDetectionTrace(
        detector_description=ChangePointDetectorDescription(name="detector"),
        detected_change_points=[],
        threshold=None,
        processing_time=np.zeros(len(values), dtype=np.float64),
        detection_function=values,
        algorithm_states=[CountingAlgorithmState() for _ in range(len(values))],
    )


def _make_bisegment_run(values: UnivariateNumericArray, change_point: int) -> SingleRun:
    trace = _make_trace(values)
    annotation = _make_annotation(provider_type="bisegment")
    provider = _make_provider(annotation=annotation, change_points=(change_point,))
    return SingleRun(trace=trace, provider=provider)  # type: ignore[arg-type]
