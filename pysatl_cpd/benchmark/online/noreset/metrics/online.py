# -*- coding: ascii -*-
"""No-reset online metrics (ARL and delay metrics)."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

from pysatl_cpd.analysis.metrics.multiple_run.online.arl import ARLMetric
from pysatl_cpd.analysis.metrics.multiple_run.online.delay import MeanDelayMetric, MedianDelayMetric
from pysatl_cpd.benchmark.online.noreset.detector.noreset_trace import NoResetDetectionTrace
from pysatl_cpd.benchmark.online.noreset.metrics.base import NoResetMultipleRunMetric
from pysatl_cpd.benchmark.online.noreset.metrics.policy import NoChangePolicy
from pysatl_cpd.benchmark.online.noreset.metrics.policy.bisegment import MixedPolicy
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState
from pysatl_cpd.data.providers.labeled import LabeledData


class NoResetARLMetric[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](
    NoResetMultipleRunMetric[StateT, ProviderT, float]
):
    """No-reset ARL metric using the no-change policy.

    Parameters
    ----------
    strict
        Whether to use strict inequality when comparing detection
        function values against the threshold (default True).
    """

    def __init__(self, *, strict: bool = True) -> None:
        super().__init__(
            source=ARLMetric[NoResetDetectionTrace[StateT], ProviderT](),
            policy=NoChangePolicy(strict=strict),
        )


class NoResetMeanDelayMetric[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](
    NoResetMultipleRunMetric[StateT, ProviderT, float]
):
    """No-reset mean delay metric using mixed policy semantics.

    Parameters
    ----------
    max_delay
        Maximum allowed delay (in steps) for a true positive.
    strict
        Whether to use strict inequality when comparing detection
        function values against the threshold (default True).
    """

    def __init__(self, *, max_delay: int, strict: bool = True) -> None:
        super().__init__(
            source=MeanDelayMetric[NoResetDetectionTrace[StateT], ProviderT](max_delay),
            policy=MixedPolicy(max_delay=max_delay, strict=strict),
        )


class NoResetMedianDelayMetric[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](
    NoResetMultipleRunMetric[StateT, ProviderT, float]
):
    """No-reset median delay metric using mixed policy semantics.

    Parameters
    ----------
    max_delay
        Maximum allowed delay (in steps) for a true positive.
    strict
        Whether to use strict inequality when comparing detection
        function values against the threshold (default True).
    """

    def __init__(self, *, max_delay: int, strict: bool = True) -> None:
        super().__init__(
            source=MedianDelayMetric[NoResetDetectionTrace[StateT], ProviderT](max_delay),
            policy=MixedPolicy(max_delay=max_delay, strict=strict),
        )
