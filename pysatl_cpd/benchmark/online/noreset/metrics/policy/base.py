# -*- coding: ascii -*-
"""No-reset policy protocol and shared helpers."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from pysatl_cpd.benchmark.online.noreset.detector.noreset_trace import NoResetDetectionTrace
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data.providers.labeled import LabeledData
from pysatl_cpd.data.typedefs import ProviderType
from pysatl_cpd.typedefs import BoolArray, UnivariateNumericArray


class NoResetPolicy[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](Protocol):
    """Transforms an infinite-threshold run into a threshold-marked run."""

    def apply(  # pragma: no cover
        self,
        run: SingleRun[OnlineDetectionTrace[StateT], ProviderT],
        threshold: float,
    ) -> SingleRun[NoResetDetectionTrace[StateT], ProviderT]:
        """Apply the policy to a run at a specific threshold.

        Parameters
        ----------
        run
            Input run with online detection trace.
        threshold
            Detection threshold.

        Returns
        -------
        SingleRun[NoResetDetectionTrace, ProviderT]
        """
        ...


def _point_mask(values: UnivariateNumericArray, threshold: float, strict: bool) -> BoolArray:
    """Return a boolean mask of threshold-satisfying points.

    Parameters
    ----------
    values
        Detection function values.
    threshold
        Threshold value.
    strict
        Use strict (> vs >=) comparison.

    Returns
    -------
    BoolArray
    """
    return np.greater(values, threshold) if strict else np.greater_equal(values, threshold)


def _event_mask(values: UnivariateNumericArray, threshold: float, strict: bool) -> BoolArray:
    """Return a boolean mask of threshold-crossing events.

    An event is the first point at or above threshold after being below.

    Parameters
    ----------
    values
        Detection function values.
    threshold
        Threshold value.
    strict
        Use strict (> vs >=) comparison.

    Returns
    -------
    BoolArray
    """
    if len(values) == 0:
        return np.zeros(0, dtype=np.bool_)

    curr_mask = _point_mask(values, threshold, strict)
    prev_mask = np.concatenate((np.array([True], dtype=np.bool_), np.less_equal(values[:-1], threshold)))
    return prev_mask & curr_mask


def _build_noreset_run[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](
    run: SingleRun[OnlineDetectionTrace[StateT], ProviderT],
    threshold: float,
    detected_change_points: Sequence[int],
) -> SingleRun[NoResetDetectionTrace[StateT], ProviderT]:
    """Construct a no-reset run with policy-selected detections.

    Parameters
    ----------
    run
        Input run.
    threshold
        Applied threshold value.
    detected_change_points
        Policy-selected detection indices.

    Returns
    -------
    SingleRun[NoResetDetectionTrace, ProviderT]
    """
    normalized_points = sorted(set(detected_change_points))
    return SingleRun(
        trace=NoResetDetectionTrace.from_inf_trace(
            source_trace=run.trace,
            detected_change_points=normalized_points,
            threshold=threshold,
        ),
        provider=run.provider,
    )


def _validate_no_change_run[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](
    run: SingleRun[OnlineDetectionTrace[StateT], ProviderT],
) -> None:
    """Ensure the run is suitable for no-change policy evaluation.

    Parameters
    ----------
    run
        Input run to validate.

    Raises
    ------
    ValueError
        If the provider is not a no-change provider.
    """
    if getattr(run.provider.annotation, "provider_type", None) != ProviderType.NO_CHANGE:
        raise ValueError("No-reset no-change policies require no-change providers")


def _validate_bisegment_run[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](
    run: SingleRun[OnlineDetectionTrace[StateT], ProviderT],
) -> int:
    """Ensure the run is a single-change bisegment and return that change point.

    Parameters
    ----------
    run
        Input run to validate.

    Returns
    -------
    int
        The single true change point index.

    Raises
    ------
    ValueError
        If the run is not a bisegment or has != 1 change point.
    """
    if getattr(run.provider.annotation, "provider_type", None) != "bisegment":
        raise ValueError("No-reset classification policies require bisegment providers")
    if len(run.provider.change_points) != 1:
        raise ValueError("No-reset classification policies require exactly one true change point")
    return run.provider.change_points[0]


def _region_points(mask: BoolArray, start: int, end: int) -> list[int]:
    """Collect all marked points in an inclusive region.

    Parameters
    ----------
    mask
        Boolean mask.
    start
        Start index (inclusive).
    end
        End index (inclusive).

    Returns
    -------
    list[int]
    """
    if start > end:
        return []
    return (np.flatnonzero(mask[start : end + 1]) + start).tolist()


def _first_region_point(mask: BoolArray, start: int, end: int) -> list[int]:
    """Collect the first marked point in an inclusive region.

    Parameters
    ----------
    mask
        Boolean mask.
    start
        Start index (inclusive).
    end
        End index (inclusive).

    Returns
    -------
    list[int]
    """
    points = _region_points(mask, start, end)
    return points[:1]
