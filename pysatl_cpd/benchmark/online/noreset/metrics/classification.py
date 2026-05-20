# -*- coding: ascii -*-
"""No-reset classification metrics (TP, FP, FN, precision, recall, F1, report)."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

from pysatl_cpd.analysis.metrics.multiple_run.classification import TotalFN, TotalFP, TotalTP
from pysatl_cpd.analysis.metrics.multiple_run.classification.fmeasure import FScoreMetric
from pysatl_cpd.analysis.metrics.multiple_run.classification.precision import PrecisionMetric
from pysatl_cpd.analysis.metrics.multiple_run.classification.recall import RecallMetric
from pysatl_cpd.analysis.metrics.multiple_run.classification.report import ClassificationReport
from pysatl_cpd.benchmark.online.noreset.detector.noreset_trace import NoResetDetectionTrace
from pysatl_cpd.benchmark.online.noreset.metrics.base import NoResetDerivedMetric, NoResetMultipleRunMetric
from pysatl_cpd.benchmark.online.noreset.metrics.policy.bisegment import BisegmentPolicyBase
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState
from pysatl_cpd.data.providers.labeled import LabeledData
from pysatl_cpd.typedefs import Number


class NoResetTotalTPMetric[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](
    NoResetMultipleRunMetric[StateT, ProviderT, int]
):
    """No-reset total true positive metric.

    Parameters
    ----------
    error_margin
        (Left, right) tolerance around the true change point.
    policy
        Bisegment policy defining true-region detection rules.
    """

    def __init__(self, *, error_margin: tuple[int, int], policy: BisegmentPolicyBase[StateT, ProviderT]) -> None:
        super().__init__(
            source=TotalTP[NoResetDetectionTrace[StateT], ProviderT](error_margin),
            policy=policy,
        )


class NoResetTotalFPMetric[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](
    NoResetMultipleRunMetric[StateT, ProviderT, int]
):
    """No-reset total false positive metric.

    Parameters
    ----------
    error_margin
        (Left, right) tolerance around the true change point.
    policy
        Bisegment policy defining false-region detection rules.
    """

    def __init__(self, *, error_margin: tuple[int, int], policy: BisegmentPolicyBase[StateT, ProviderT]) -> None:
        super().__init__(
            source=TotalFP[NoResetDetectionTrace[StateT], ProviderT](error_margin),
            policy=policy,
        )


class NoResetTotalFNMetric[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](
    NoResetMultipleRunMetric[StateT, ProviderT, int]
):
    """No-reset total false negative metric.

    Parameters
    ----------
    error_margin
        (Left, right) tolerance around the true change point.
    policy
        Bisegment policy defining true-region detection rules.
    """

    def __init__(self, *, error_margin: tuple[int, int], policy: BisegmentPolicyBase[StateT, ProviderT]) -> None:
        super().__init__(
            source=TotalFN[NoResetDetectionTrace[StateT], ProviderT](error_margin),
            policy=policy,
        )


class NoResetPrecisionMetric[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](
    NoResetDerivedMetric[StateT, ProviderT, Number, float]
):
    """No-reset precision metric with independently configurable TP/FP policies.

    Parameters
    ----------
    error_margin
        (Left, right) tolerance around the true change point.
    tp_policy
        Policy used for the true-positive source metric.
    fp_policy
        Policy used for the false-positive source metric.
    """

    def __init__(
        self,
        *,
        error_margin: tuple[int, int],
        tp_policy: BisegmentPolicyBase[StateT, ProviderT],
        fp_policy: BisegmentPolicyBase[StateT, ProviderT],
    ) -> None:
        super().__init__(
            source=PrecisionMetric[NoResetDetectionTrace[StateT], ProviderT](error_margin),
            bases={
                "tp": NoResetTotalTPMetric(error_margin=error_margin, policy=tp_policy),
                "fp": NoResetTotalFPMetric(error_margin=error_margin, policy=fp_policy),
            },
        )


class NoResetRecallMetric[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](
    NoResetDerivedMetric[StateT, ProviderT, Number, float]
):
    """No-reset recall metric with independently configurable TP/FN policies.

    Parameters
    ----------
    error_margin
        (Left, right) tolerance around the true change point.
    tp_policy
        Policy used for the true-positive source metric.
    fn_policy
        Policy used for the false-negative source metric.
    """

    def __init__(
        self,
        *,
        error_margin: tuple[int, int],
        tp_policy: BisegmentPolicyBase[StateT, ProviderT],
        fn_policy: BisegmentPolicyBase[StateT, ProviderT],
    ) -> None:
        super().__init__(
            source=RecallMetric[NoResetDetectionTrace[StateT], ProviderT](error_margin),
            bases={
                "tp": NoResetTotalTPMetric(error_margin=error_margin, policy=tp_policy),
                "fn": NoResetTotalFNMetric(error_margin=error_margin, policy=fn_policy),
            },
        )


class NoResetF1Metric[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](
    NoResetDerivedMetric[StateT, ProviderT, float, float]
):
    """No-reset F1 metric derived from no-reset precision and recall metrics.

    Parameters
    ----------
    error_margin
        (Left, right) tolerance around the true change point.
    precision_metric
        Pre-configured no-reset precision metric.
    recall_metric
        Pre-configured no-reset recall metric.
    """

    def __init__(
        self,
        *,
        error_margin: tuple[int, int],
        precision_metric: NoResetPrecisionMetric[StateT, ProviderT],
        recall_metric: NoResetRecallMetric[StateT, ProviderT],
    ) -> None:
        super().__init__(
            source=FScoreMetric[NoResetDetectionTrace[StateT], ProviderT](error_margin),
            bases={
                "precision": precision_metric,
                "recall": recall_metric,
            },
        )


class NoResetClassificationReport[StateT: OnlineAlgorithmState, ProviderT: LabeledData[Any, Any]](
    NoResetDerivedMetric[StateT, ProviderT, Number, dict[str, Number]]
):
    """No-reset classification report with one global policy and optional policy overrides.

    Parameters
    ----------
    error_margin
        (Left, right) tolerance around the true change point.
    global_policy
        Default policy applied to all source metrics.
    precision_policy
        Optional override policy for precision and its TP/FP bases.
    recall_policy
        Optional override policy for recall and its TP/FN bases.
    """

    def __init__(
        self,
        *,
        error_margin: tuple[int, int],
        global_policy: BisegmentPolicyBase[StateT, ProviderT],
        precision_policy: BisegmentPolicyBase[StateT, ProviderT] | None = None,
        recall_policy: BisegmentPolicyBase[StateT, ProviderT] | None = None,
    ) -> None:
        precision_metric = NoResetPrecisionMetric(
            error_margin=error_margin,
            tp_policy=precision_policy or global_policy,
            fp_policy=precision_policy or global_policy,
        )
        recall_metric = NoResetRecallMetric(
            error_margin=error_margin,
            tp_policy=recall_policy or global_policy,
            fn_policy=recall_policy or global_policy,
        )
        super().__init__(
            source=ClassificationReport[NoResetDetectionTrace[StateT], ProviderT](error_margin),
            bases={
                "tp": NoResetTotalTPMetric(error_margin=error_margin, policy=global_policy),
                "fp": NoResetTotalFPMetric(error_margin=error_margin, policy=global_policy),
                "fn": NoResetTotalFNMetric(error_margin=error_margin, policy=global_policy),
                "precision": precision_metric,
                "recall": recall_metric,
                "f1": NoResetF1Metric(
                    error_margin=error_margin,
                    precision_metric=precision_metric,
                    recall_metric=recall_metric,
                ),
            },
        )
