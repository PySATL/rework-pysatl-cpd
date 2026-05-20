# -*- coding: ascii -*-
"""
No-reset online change-point detector.

Runs the same loop as :class:`OnlineResetDetector` but does **not** call
:meth:`OnlineAlgorithm.reset` when a changepoint is declared (the detection
statistic is not restarted), while still applying skip periods and run-length
rules from the solver.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from pysatl_cpd.benchmark.online.noreset.tooling.bisegment_cut import NOOP_BISEGMENT_CUT, BisegmentCut
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.online import OnlineDetector
from pysatl_cpd.core.online.detectors.online_detection_trace import (
    OnlineDetectionStepResult,
    OnlineDetectionTrace,
)
from pysatl_cpd.core.online.ionline_algorithm import (
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
)
from pysatl_cpd.data.providers.data_provider import DataProvider
from pysatl_cpd.data.providers.transformers.base import IDataTransformer
from pysatl_cpd.data.typedefs import TimeseriesAnnotation
from pysatl_cpd.typedefs import Number, frozendict

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


class NoResetOnlineDetector[
    DataT,
    ConfigurationT: OnlineAlgorithmConfiguration,
    StateT: OnlineAlgorithmState,
](OnlineDetector[DataT, ConfigurationT, StateT]):
    """
    Online detector that does not reset the algorithm after a changepoint.

    For the reset loop, see :class:`OnlineResetDetector`; the only difference is
    that the post-detection :meth:`OnlineAlgorithm.reset` call is omitted.

    Parameters
    ----------
    algorithm
        Algorithm instance that drives the per-step detection logic.
    collect_states
        Whether to retain algorithm states at every step (default True).
    data_transformer
        Optional transformer applied to incoming data before processing.
    bisegment_cut
        Optional provider crop margins (left/right in sample points).
    """

    def __init__(
        self,
        algorithm: OnlineAlgorithm[DataT, ConfigurationT, StateT],
        *,
        collect_states: bool = True,
        data_transformer: IDataTransformer[Any, Any] | None = None,
        bisegment_cut: BisegmentCut = NOOP_BISEGMENT_CUT,
    ) -> None:
        super().__init__(algorithm, data_transformer=data_transformer)
        self._collect_states = collect_states
        self._bisegment_cut = bisegment_cut

    @property
    def description(self) -> ChangePointDetectorDescription:
        """Description identifying this detector as a no-reset online detector."""
        return ChangePointDetectorDescription(
            name="no_reset_online_detector",
            parameters=frozendict(
                algorithm=self.algorithm.description,
                data_transformer=self.data_transformer.annotation if self.data_transformer is not None else None,
                left_trim=self.left_trim,
                right_trim=self.right_trim,
            ),
        )

    @property
    def left_trim(self) -> int:
        """Left trim in sample points used by detector-level crop."""
        return self._bisegment_cut.left_trim

    @property
    def right_trim(self) -> int:
        """Right trim in sample points used by detector-level crop."""
        return self._bisegment_cut.right_trim

    def _detect[AnnotationT: TimeseriesAnnotation](
        self, data: DataProvider[DataT, AnnotationT]
    ) -> OnlineDetectionTrace[StateT]:
        """Run the no-reset detection loop over a data provider.

        Delegates to the private ``_run_no_reset`` generator and packs
        the yielded steps into an ``OnlineDetectionTrace`` with no
        threshold filtering.

        Parameters
        ----------
        data
            Provider that yields observations sequentially.

        Returns
        -------
        OnlineDetectionTrace
            Trace containing all detection steps with ``threshold`` set
            to None.
        """
        if self._bisegment_cut.is_noop:
            steps: list[OnlineDetectionStepResult[StateT]] = list(
                _run_no_reset(self.algorithm, data, self._collect_states)
            )
            return OnlineDetectionTrace.from_run(
                steps=steps,
                detector_description=self.description,
                threshold=None,
            )

        total_length = len(data)
        self._bisegment_cut.validate_for_length(total_length)
        start = self.left_trim
        stop = total_length - self.right_trim - 1
        cropped = data.cut(start, stop)
        cropped_steps: list[OnlineDetectionStepResult[StateT]] = list(
            _run_no_reset(self.algorithm, cropped, self._collect_states)
        )
        steps = self._pad_trimmed_steps(cropped_steps)
        return OnlineDetectionTrace.from_run(
            steps=steps,
            detector_description=self.description,
            threshold=None,
        )

    def clone(self) -> NoResetOnlineDetector[DataT, ConfigurationT, StateT]:
        """Create an independent copy of this detector.

        Reconstructs the detector by calling ``recreate()`` on the
        underlying algorithm, ensuring parallel workers each have
        an isolated instance.

        Returns
        -------
        NoResetOnlineDetector
            A new detector instance with identical configuration.
        """
        return NoResetOnlineDetector(
            self.algorithm.recreate(),
            collect_states=self._collect_states,
            data_transformer=self.data_transformer,
            bisegment_cut=self._bisegment_cut,
        )

    def _pad_trimmed_steps(
        self,
        cropped_steps: list[OnlineDetectionStepResult[StateT]],
    ) -> list[OnlineDetectionStepResult[StateT]]:
        """Restore original provider length by adding skip-only trim steps."""
        prefix = [self._make_skip_step(step_num=i) for i in range(self.left_trim)]
        shifted = [replace(step, step_num=step.step_num + self.left_trim) for step in cropped_steps]
        suffix_start = self.left_trim + len(cropped_steps)
        suffix = [self._make_skip_step(step_num=suffix_start + i) for i in range(self.right_trim)]
        return [*prefix, *shifted, *suffix]

    def _make_skip_step(self, step_num: int) -> OnlineDetectionStepResult[StateT]:
        """Build a skip-only step used for trim padding."""
        return OnlineDetectionStepResult(
            step_num=step_num,
            is_in_skip_period=True,
        )


def _run_no_reset[
    DataT,
    AnnotationT: TimeseriesAnnotation,
    ConfigurationT: OnlineAlgorithmConfiguration,
    StateT: OnlineAlgorithmState,
](
    algorithm: OnlineAlgorithm[DataT, ConfigurationT, StateT],
    data_provider: DataProvider[DataT, AnnotationT],
    collect_states: bool,
) -> Iterator[OnlineDetectionStepResult[StateT]]:
    """Core no-reset detection generator.

    Resets the algorithm once, then iterates over data observations
    without issuing a reset after detections. Yields a step result
    for every observation.

    Parameters
    ----------
    algorithm
        Algorithm instance driving per-step detection.
    data_provider
        Source of sequential observations.
    collect_states
        If True, includes the algorithm state in each yielded step.

    Yields
    ------
    OnlineDetectionStepResult
        One result per observation step.
    """
    algorithm.reset()

    algorithm.reset()
    for step, observation in enumerate(data_provider):
        step_start_time: float = time.perf_counter()
        detection_func: Number = algorithm.process(observation)
        step_finish_time: float = time.perf_counter()

        algorithm_state = algorithm.state if collect_states else None

        yield OnlineDetectionStepResult(
            step_num=step,
            is_in_skip_period=False,
            is_forced_change_point=False,
            is_signal_change_point=False,
            detection_function=detection_func,
            processing_time=step_finish_time - step_start_time,
            algorithm_state=algorithm_state,
        )
