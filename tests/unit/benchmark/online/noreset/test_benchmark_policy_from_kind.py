# -*- coding: ascii -*-
"""
Tests for benchmark policy from kind.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.benchmark.online.noreset.benchmark import OnlineNoResetBenchmark
from pysatl_cpd.benchmark.online.noreset.metrics.policy.bisegment import (
    EventBasedPolicy,
    MixedPolicy,
    PointBasedPolicy,
)
from pysatl_cpd.benchmark.online.noreset.policy_kind import NoResetPolicyKind


class TestPolicyFromKind:
    def test_point_policy(self) -> None:
        policy = OnlineNoResetBenchmark._policy_from_kind(NoResetPolicyKind.POINT, max_delay=5, strict=True)
        assert isinstance(policy, PointBasedPolicy)

    def test_event_policy(self) -> None:
        policy = OnlineNoResetBenchmark._policy_from_kind(NoResetPolicyKind.EVENT, max_delay=5, strict=False)
        assert isinstance(policy, EventBasedPolicy)

    def test_mixed_policy(self) -> None:
        policy = OnlineNoResetBenchmark._policy_from_kind(NoResetPolicyKind.MIXED, max_delay=5, strict=True)
        assert isinstance(policy, MixedPolicy)
