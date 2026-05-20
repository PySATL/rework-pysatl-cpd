# -*- coding: ascii -*-
"""
Online reset change-point detector.

This module implements a detector that resets the online algorithm after each
declared change point.
"""

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import time
from collections.abc import Iterator
from typing import Any, Self, cast

from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionStepResult, OnlineDetectionTrace
from pysatl_cpd.core.online.detectors.online_detector import OnlineDetector
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithm, OnlineAlgorithmConfiguration, OnlineAlgorithmState
from pysatl_cpd.data import DataProvider
from pysatl_cpd.data.providers.transformers.base import IDataTransformer
from pysatl_cpd.data.typedefs import TimeseriesAnnotation
from pysatl_cpd.typedefs import Number, frozendict

# TODO: refactor and remove being generic over annotation - move to detect methods


class OnlineResetDetector[
    DataT,
    ConfigurationT: OnlineAlgorithmConfiguration,
    StateT: OnlineAlgorithmState,
](OnlineDetector[DataT, ConfigurationT, StateT]):
    """
    Online detector that resets its algorithm after every detected change point.

    Iterates over a data provider, feeds each observation to the algorithm,
    compares the resulting statistic against a threshold, and optionally
    enforces a maximum run-length and a post-detection skip period.

    Parameters
    ----------
    algorithm
        Algorithm instance that drives per-step detection.
    threshold
        Detection threshold. A change point is declared when the
        algorithm's detection statistic exceeds this value.
        Default is ``nan`` (no signal-based detections).
    skip_period
        Number of steps to skip (suppress detections) after each declared
        change point. Must be non-negative.
    max_runlength
        If not ``None``, forces a change-point declaration once the run
        length exceeds this value. Must be positive if specified.
    collect_states
        Whether to collect algorithm state snapshots in step results.
        If False, algorithm_state will be None in all step results.
    data_transformer
        Optional transformer applied to incoming data before detection.

    Raises
    ------
    ValueError
        If ``skip_period`` is negative, or if ``max_runlength`` is not
        None and not positive.
    """

    def __init__(
        self,
        algorithm: OnlineAlgorithm[DataT, ConfigurationT, StateT],
        *,
        threshold: float = float("nan"),
        skip_period: int = 0,
        max_runlength: int | None = None,
        collect_states: bool = True,
        data_transformer: IDataTransformer[Any, Any] | None = None,
    ) -> None:
        # Validate skip_period is non-negative
        if skip_period < 0:
            raise ValueError(f"skip_period must be non-negative, got {skip_period}")

        # Validate max_runlength is positive if specified
        if max_runlength is not None and max_runlength <= 0:
            raise ValueError(f"max_runlength must be positive if specified, got {max_runlength}")

        super().__init__(algorithm, data_transformer=data_transformer)
        self.threshold = threshold
        self.skip_period = skip_period
        self.max_runlength = max_runlength
        self.collect_states = collect_states

    @property
    def description(self) -> ChangePointDetectorDescription:
        """Return a static description of this detector and its parameters.

        Returns
        -------
        ChangePointDetectorDescription
            Description containing the detector name, threshold, skip
            period, max run-length, state-collection flag, and data
            transformer annotation.
        """
        return ChangePointDetectorDescription(
            name="online_reset_detector",
            parameters=frozendict(
                algorithm=self.algorithm.description,
                threshold=self.threshold,
                skip_period=self.skip_period,
                max_runlength=self.max_runlength,
                collect_states=self.collect_states,
                data_transformer=self.data_transformer.annotation if self.data_transformer is not None else None,
            ),
        )

    def _detect[AnnotationT: TimeseriesAnnotation](
        self, data: DataProvider[DataT, AnnotationT]
    ) -> OnlineDetectionTrace[StateT]:
        """Run detection over the complete data provider.

        Parameters
        ----------
        data
            Source of observations to process.

        Returns
        -------
        OnlineDetectionTrace
            Trace containing all per-step detection results.
        """
        steps = list(self._run(data))
        return OnlineDetectionTrace.from_run(
            steps=steps,
            detector_description=self.description,
            threshold=self.threshold,
        )

    def clone(self) -> Self:
        """Create an independent copy of this detector.

        The clone uses a recreated (fresh) algorithm instance and shares the
        same threshold, skip period, max run-length, and data transformer
        settings as the original.

        Returns
        -------
        Self
            A new detector instance with identical configuration.
        """
        return cast(
            Self,
            OnlineResetDetector(
                self.algorithm.recreate(),
                threshold=self.threshold,
                skip_period=self.skip_period,
                max_runlength=self.max_runlength,
                collect_states=self.collect_states,
                data_transformer=self.data_transformer,
            ),
        )

    def _run[AnnotationT: TimeseriesAnnotation](
        self,
        data_provider: DataProvider[DataT, AnnotationT],
    ) -> Iterator[OnlineDetectionStepResult[StateT]]:
        """
        Execute the detection loop over all observations.

        Iterates through all observations provided by the data provider,
        processes each through the detection algorithm, and yields per-step
        results. During a skip period following a detected change point,
        observations are processed without change point declarations.

        Parameters
        ----------
        data_provider
            An iterable source of observations.

        Yields
        ------
        OnlineDetectionStepResult
            Per-step detection result containing the change-point flag,
            statistic value, step index, processing time, and algorithm state.

        Notes
        -----
        The solver maintains three key state variables:
        - run_length: Number of observations since the last change point
        - skip_period_counter: Number of steps remaining in skip period
        - in_skip_period: Flag indicating active skip period
        """
        run_length: int = 0
        skip_period_counter: int = 0
        in_skip_period: bool = False

        self.algorithm.reset()

        for step, observation in enumerate(data_provider):
            # Handle skip period where detections are suppressed
            if in_skip_period:
                if skip_period_counter < self.skip_period:
                    skip_period_counter += 1
                    yield OnlineDetectionStepResult(
                        step_num=step,
                        is_in_skip_period=True,
                        algorithm_state=None,
                    )
                    continue
                in_skip_period = False
                skip_period_counter = 0

            # Process observation normally
            step_start_time: float = time.perf_counter()
            detection_func: Number = self.algorithm.process(observation)
            step_finish_time: float = time.perf_counter()

            run_length += 1

            # Determine if change point occurred
            is_forced: bool = self._is_forced_change_point(run_length)
            is_signal: bool = self._is_signal_change_point(detection_func, self.threshold)
            is_change_point: bool = is_forced | is_signal

            # Get algorithm state if collecting
            algorithm_state = self.algorithm.state if self.collect_states else None

            yield OnlineDetectionStepResult(
                step_num=step,
                is_in_skip_period=False,
                is_forced_change_point=is_forced,
                is_signal_change_point=is_signal,
                detection_function=detection_func,
                processing_time=step_finish_time - step_start_time,
                algorithm_state=algorithm_state,
            )

            # Handle change point detection
            if is_change_point:
                self.algorithm.reset()
                in_skip_period = True
                run_length = 0

    def _is_signal_change_point(self, detection_func: Number, threshold: float) -> bool:
        """
        Determine if detection statistic exceeds threshold.

        Parameters
        ----------
        detection_func
            Current detection statistic value.
        threshold
            Detection threshold for the change-point function.

        Returns
        -------
        bool
            True if statistic exceeds threshold, False otherwise.
        """
        return bool(detection_func > threshold)

    def _is_forced_change_point(self, run_length: int) -> bool:
        """
        Determine if forced change point is required due to run length.

        Parameters
        ----------
        run_length
            Number of observations since last change point.

        Returns
        -------
        bool
            True if run length exceeds max_runlength, False otherwise.
        """
        return bool(self.max_runlength is not None and run_length > self.max_runlength)
