# -*- coding: ascii -*-
"""No-reset classification policies for continuous detector traces.

In no-reset benchmarking the detector runs continuously without restarting
after each alarm.  A *policy* decides which thresholded points on that
continuous detection function count as true positives, false positives, or
false negatives around a true change point (or inside a no-change regime).

Policies do not modify the underlying trace.  They translate an
infinite-threshold run into a threshold-marked ``NoResetDetectionTrace``
that downstream metrics (precision, recall, F1, ARL, ...) can consume.

.. raw:: html

    <h2>Public API</h2>

- ``NoResetPolicy`` -- protocol that all policies implement.  The core
  method is ``apply(run, threshold) -> SingleRun[NoResetDetectionTrace, ProviderT]``.
- ``BisegmentPolicyBase`` -- abstract base for policies that operate on
  single-change-point bisegment providers.  Subclasses define how to
  select detections in the false region (before the change point) and the
  true region (at or after the change point).
- ``PointBasedPolicy`` -- treats every threshold-satisfying point as a
  candidate detection.  In the true region only the first such point is
  kept.
- ``EventBasedPolicy`` -- focuses on threshold-crossing *events* (the
  first point at or above threshold after being below).  In the true
  region only the first event is kept.
- ``MixedPolicy`` -- uses event-based selection in the false region and
  point-based selection in the true region.
- ``NoChangePolicy`` -- for no-change (ARL) providers.  Keeps at most
  the first point that satisfies the threshold.

See the docstrings in ``base``, ``bisegment``, and ``nochange`` submodules
for implementation details of each class.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Create a policy and apply it to a single run::

    >>> from pysatl_cpd.benchmark.online.noreset.metrics.policy import (
    ...     MixedPolicy,
    ...     PointBasedPolicy,
    ...     EventBasedPolicy,
    ...     NoChangePolicy,
    ... )
    >>> point_policy = PointBasedPolicy(max_delay=15, strict=False)
    >>> event_policy = EventBasedPolicy(max_delay=15, strict=False)
    >>> mixed_policy = MixedPolicy(max_delay=15, strict=False)
    >>> nochange_policy = NoChangePolicy(strict=False)

Policies are typically consumed through ``OnlineNoResetBenchmark`` rather
than called directly.  The benchmark constructs the appropriate policy
from a ``NoResetPolicyKind``::

    >>> from pysatl_cpd.benchmark.online.noreset import (
    ...     NoResetPolicyKind,
    ...     OnlineNoResetBenchmark,
    ... )
    >>> # benchmark = OnlineNoResetBenchmark(
    ... #     dataset=dataset,
    ... #     registry=registry,
    ... #     max_delay=15,
    ... #     global_policy=NoResetPolicyKind.MIXED,
    ... #     error_margin=(0, 15),
    ... #     policy_strict=False,
    ... # )

.. raw:: html

    <h2>Notes</h2>

Notes
-----
- All bisegment policies (``PointBasedPolicy``, ``EventBasedPolicy``,
  ``MixedPolicy``) require the run provider to be a bisegment with
  exactly one true change point.
- ``NoChangePolicy`` requires the run provider to be a no-change provider
  (``provider_type == ProviderType.NO_CHANGE``).
- The ``strict`` parameter controls whether threshold comparison uses
  strict (``>``) or non-strict (``>=``) inequality.
- ``max_delay`` must be non-negative and defines the size of the true
  region as ``[cp, min(cp + max_delay, length - 1)]``.
- Policies are stateless after construction and can be reused across
  multiple runs and thresholds.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.benchmark.online.noreset.metrics.policy.base import NoResetPolicy
from pysatl_cpd.benchmark.online.noreset.metrics.policy.bisegment import (
    BisegmentPolicyBase,
    EventBasedPolicy,
    MixedPolicy,
    PointBasedPolicy,
)
from pysatl_cpd.benchmark.online.noreset.metrics.policy.nochange import NoChangePolicy

__all__ = [
    "BisegmentPolicyBase",
    "EventBasedPolicy",
    "MixedPolicy",
    "NoChangePolicy",
    "NoResetPolicy",
    "PointBasedPolicy",
]
