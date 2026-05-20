# -*- coding: ascii -*-
"""Bisegment-based policies for no-reset classification evaluation."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from typing import Any

from pysatl_cpd.benchmark.online.noreset.detector.noreset_trace import NoResetDetectionTrace
from pysatl_cpd.benchmark.online.noreset.metrics.policy.base import (
    _build_noreset_run,
    _event_mask,
    _first_region_point,
    _point_mask,
    _region_points,
    _validate_bisegment_run,
)
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.providers.labeled import LabeledData
from pysatl_cpd.typedefs import UnivariateNumericArray


class BisegmentPolicyBase[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](ABC):
    """Shared implementation for no-reset bisegment policies.

    Parameters
    ----------
    max_delay
        Maximum allowed delay (in steps) for a detection to be
        considered a true positive. Must be non-negative.
    strict
        Whether to use strict inequality when comparing detection
        function values against the threshold (default True).

    Raises
    ------
    ValueError
        If ``max_delay`` is negative.
    """

    def __init__(self, *, max_delay: int, strict: bool = True) -> None:
        if max_delay < 0:
            raise ValueError("max_delay must be non-negative")
        self._max_delay = max_delay
        self._strict = strict

    @abstractmethod
    def _select_false_region(  # pragma: no cover
        self,
        values: UnivariateNumericArray,
        threshold: float,
        cp: int,
    ) -> list[int]:
        """Select detections in the false region (before change point).

        Parameters
        ----------
        values
            Detection function values.
        threshold
            Detection threshold.
        cp
            True change point index.

        Returns
        -------
        list[int]
        """
        ...

    @abstractmethod
    def _select_true_region(  # pragma: no cover
        self,
        values: UnivariateNumericArray,
        threshold: float,
        cp: int,
    ) -> list[int]:
        """Select detections in the true region (at or after change point).

        Parameters
        ----------
        values
            Detection function values.
        threshold
            Detection threshold.
        cp
            True change point index.

        Returns
        -------
        list[int]
        """
        ...

    def apply(
        self,
        run: SingleRun[OnlineDetectionTrace[StateT], ProviderT],
        threshold: float,
    ) -> SingleRun[NoResetDetectionTrace[StateT], ProviderT]:
        """Apply the policy to a single run and return a classified trace.

        Validates the run as a bisegment run, computes detection points
        in both the false and true regions, and packages the result into
        a ``NoResetDetectionTrace``.

        Parameters
        ----------
        run
            Input run with an ``OnlineDetectionTrace`` and labeled data.
        threshold
            Threshold applied to the detection function values.

        Returns
        -------
        SingleRun[NoResetDetectionTrace, ProviderT]
            Run wrapping a classified no-reset trace.
        """
        cp = _validate_bisegment_run(run)
        values = run.trace.detection_function
        points = [
            *self._select_false_region(values, threshold, cp),
            *self._select_true_region(values, threshold, cp),
        ]
        return _build_noreset_run(run, threshold, points)

    def _true_region_end(self, cp: int, length: int) -> int:
        """Return the inclusive end of the true-change region.

        Parameters
        ----------
        cp
            True change point index.
        length
            Total length of the detection function.

        Returns
        -------
        int
        """
        return min(cp + self._max_delay, length - 1)


class PointBasedPolicy[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](
    BisegmentPolicyBase[StateT, ProviderT]
):
    """Point-based no-reset policy."""

    def _select_false_region(
        self,
        values: UnivariateNumericArray,
        threshold: float,
        cp: int,
    ) -> list[int]:
        """Select detections in the false region using a point mask."""
        mask = _point_mask(values, threshold, self._strict)
        return _region_points(mask, 0, cp - 1)

    def _select_true_region(
        self,
        values: UnivariateNumericArray,
        threshold: float,
        cp: int,
    ) -> list[int]:
        """Select the first detection in the true region using a point mask."""
        mask = _point_mask(values, threshold, self._strict)
        return _first_region_point(mask, cp, self._true_region_end(cp, len(values)))


class EventBasedPolicy[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](
    BisegmentPolicyBase[StateT, ProviderT]
):
    """Event-based no-reset policy."""

    def _select_false_region(
        self,
        values: UnivariateNumericArray,
        threshold: float,
        cp: int,
    ) -> list[int]:
        """Select detections in the false region using an event mask."""
        mask = _event_mask(values, threshold, self._strict)
        return _region_points(mask, 0, cp - 1)

    def _select_true_region(
        self,
        values: UnivariateNumericArray,
        threshold: float,
        cp: int,
    ) -> list[int]:
        """Select the first detection in the true region using an event mask."""
        mask = _event_mask(values, threshold, self._strict)
        return _first_region_point(mask, cp, self._true_region_end(cp, len(values)))


class MixedPolicy[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](
    BisegmentPolicyBase[StateT, ProviderT],
):
    """Event-based false region and point-based true region policy."""

    def _select_false_region(
        self,
        values: UnivariateNumericArray,
        threshold: float,
        cp: int,
    ) -> list[int]:
        """Select detections in the false region using an event mask."""
        mask = _event_mask(values, threshold, self._strict)
        return _region_points(mask, 0, cp - 1)

    def _select_true_region(
        self,
        values: UnivariateNumericArray,
        threshold: float,
        cp: int,
    ) -> list[int]:
        """Select the first detection in the true region using a point mask."""
        mask = _point_mask(values, threshold, self._strict)
        return _first_region_point(mask, cp, self._true_region_end(cp, len(values)))
