# -*- coding: ascii -*-
"""
Tests for online metrics.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.benchmark.online.noreset.metrics.online import (
    NoResetARLMetric,
    NoResetMeanDelayMetric,
    NoResetMedianDelayMetric,
)
from pysatl_cpd.benchmark.online.noreset.metrics.policy import NoChangePolicy
from pysatl_cpd.benchmark.online.noreset.metrics.policy.bisegment import MixedPolicy


class TestNoResetOnlineMetrics:
    def test_arl_metric_init(self) -> None:
        metric = NoResetARLMetric(strict=True)
        assert isinstance(metric.policy, NoChangePolicy)

    def test_arl_metric_strict_false(self) -> None:
        metric = NoResetARLMetric(strict=False)
        assert metric.policy._strict is False

    def test_mean_delay_metric_init(self) -> None:
        metric = NoResetMeanDelayMetric(max_delay=10, strict=True)
        assert isinstance(metric.policy, MixedPolicy)
        assert metric.policy._max_delay == 10

    def test_mean_delay_metric_strict_false(self) -> None:
        metric = NoResetMeanDelayMetric(max_delay=5, strict=False)
        assert metric.policy._strict is False

    def test_median_delay_metric_init(self) -> None:
        metric = NoResetMedianDelayMetric(max_delay=10, strict=True)
        assert isinstance(metric.policy, MixedPolicy)
        assert metric.policy._max_delay == 10

    def test_median_delay_metric_strict_false(self) -> None:
        metric = NoResetMedianDelayMetric(max_delay=5, strict=False)
        assert metric.policy._strict is False
