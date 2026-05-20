# -*- coding: ascii -*-
"""No-reset metric wrappers and classification policies for online CPD benchmarking.

In no-reset benchmarking the detector runs continuously without restarting after
each alarm.  This module provides wrappers that adapt classical single-run,
multiple-run, and derived metrics to the no-reset evaluation model, together
with concrete metric implementations for classification (TP, FP, FN, precision,
recall, F1, report) and online performance (ARL, mean delay, median delay).

Policies decide which thresholded points on a continuous detection function
count as true positives, false positives, or false negatives.  Metrics apply
those policies to transform raw traces into classified traces before computing
scores.  The two concerns are separated: policies live in the ``policy``
subpackage, while metric wrappers and ready-made metric classes live in this
package.

.. raw:: html

    <h2>Public API</h2>

Base wrappers
~~~~~~~~~~~~~

- ``NoResetThresholdMetric`` -- protocol for metrics that evaluate threshold
  callables over runs.
- ``NoResetSingleRunMetric`` -- wraps a classical single-run metric with a
  no-reset policy.  ``evaluate(run)`` returns ``Callable[[float], ResultT]``.
- ``NoResetMultipleRunMetric`` -- wraps a classical multiple-run metric with a
  no-reset policy.  ``evaluate(runs)`` returns ``Callable[[float], ResultT]``.
- ``NoResetDerivedMetric`` -- wraps a derived metric formula with named
  no-reset base metrics.  Raises ``ValueError`` if any required base is
  missing.
- ``wrap_noreset_single_run_metric`` -- factory for ``NoResetSingleRunMetric``.
- ``wrap_noreset_multiple_run_metric`` -- factory for ``NoResetMultipleRunMetric``.
- ``wrap_noreset_derived_metric`` -- factory for ``NoResetDerivedMetric``.

Classification metrics
~~~~~~~~~~~~~~~~~~~~~~

- ``NoResetTotalTPMetric`` -- total true positives across runs.
- ``NoResetTotalFPMetric`` -- total false positives across runs.
- ``NoResetTotalFNMetric`` -- total false negatives across runs.
- ``NoResetPrecisionMetric`` -- precision derived from TP and FP bases with
  independently configurable policies.
- ``NoResetRecallMetric`` -- recall derived from TP and FN bases with
  independently configurable policies.
- ``NoResetF1Metric`` -- F1 derived from pre-configured precision and recall
  metrics.
- ``NoResetClassificationReport`` -- full classification report (TP, FP, FN,
  precision, recall, F1) with one global policy and optional per-metric
  overrides.
- ``NoResetTPMetric`` -- alias for ``NoResetTotalTPMetric``.

Online metrics
~~~~~~~~~~~~~~

- ``NoResetARLMetric`` -- average run length using ``NoChangePolicy``.
- ``NoResetMeanDelayMetric`` -- mean detection delay using ``MixedPolicy``.
- ``NoResetMedianDelayMetric`` -- median detection delay using ``MixedPolicy``.

Policy classes
~~~~~~~~~~~~~~

- ``NoResetPolicy`` -- protocol implemented by all policies.
- ``BisegmentPolicyBase`` -- abstract base for bisegment policies.
- ``PointBasedPolicy`` -- point-based selection in both regions.
- ``EventBasedPolicy`` -- event-based selection in both regions.
- ``MixedPolicy`` -- event-based false region, point-based true region.
- ``NoChangePolicy`` -- single-detection policy for ARL evaluation.

See the ``policy`` subpackage docstring for detailed policy semantics.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Create a classification metric and evaluate it across a threshold sweep::

    >>> from pysatl_cpd.benchmark.online.noreset.metrics import (
    ...     NoResetF1Metric,
    ...     NoResetPrecisionMetric,
    ...     NoResetRecallMetric,
    ...     MixedPolicy,
    ... )
    >>> error_margin = (0, 15)
    >>> policy = MixedPolicy(max_delay=15, strict=False)
    >>> precision = NoResetPrecisionMetric(
    ...     error_margin=error_margin,
    ...     tp_policy=policy,
    ...     fp_policy=policy,
    ... )
    >>> recall = NoResetRecallMetric(
    ...     error_margin=error_margin,
    ...     tp_policy=policy,
    ...     fn_policy=policy,
    ... )
    >>> f1 = NoResetF1Metric(
    ...     error_margin=error_margin,
    ...     precision_metric=precision,
    ...     recall_metric=recall,
    ... )

Create an online metric and evaluate it::

    >>> from pysatl_cpd.benchmark.online.noreset.metrics import (
    ...     NoResetARLMetric,
    ...     NoResetMeanDelayMetric,
    ... )
    >>> arl = NoResetARLMetric(strict=False)
    >>> mean_delay = NoResetMeanDelayMetric(max_delay=15, strict=False)

Wrap a custom source metric with a no-reset policy::

    >>> from pysatl_cpd.benchmark.online.noreset.metrics import (
    ...     NoResetMultipleRunMetric,
    ...     wrap_noreset_multiple_run_metric,
    ...     MixedPolicy,
    ... )
    >>> from pysatl_cpd.analysis.metrics.multiple_run.online.arl import ARLMetric
    >>> policy = MixedPolicy(max_delay=15, strict=False)
    >>> wrapped = wrap_noreset_multiple_run_metric(ARLMetric(), policy)

.. raw:: html

    <h2>Notes</h2>

Notes
-----
- All classification metrics require bisegment providers with exactly one
  true change point.  ``NoResetARLMetric`` requires no-change providers.
- The ``error_margin`` parameter is a ``(left, right)`` tolerance tuple
  around the true change point that defines the acceptable detection window.
- The ``strict`` parameter controls threshold comparison: ``True`` uses
  strict inequality (``>``), ``False`` uses non-strict inequality (``>=``).
- Metrics return threshold-indexed evaluators (``Callable[[float], ResultT]``)
  rather than raw values, enabling efficient threshold sweeps without
  re-running the detector.
- Derived metrics (precision, recall, F1, classification report) compose
  base metrics internally; their ``evaluate`` method evaluates all bases
  at each threshold and combines results via the source formula.
- Policies are stateless after construction and safe to share across
  multiple metric instances and threshold evaluations.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.benchmark.online.noreset.metrics.base import (
    NoResetDerivedMetric,
    NoResetMultipleRunMetric,
    NoResetSingleRunMetric,
    NoResetThresholdMetric,
    wrap_noreset_derived_metric,
    wrap_noreset_multiple_run_metric,
    wrap_noreset_single_run_metric,
)
from pysatl_cpd.benchmark.online.noreset.metrics.classification import (
    NoResetClassificationReport,
    NoResetF1Metric,
    NoResetPrecisionMetric,
    NoResetRecallMetric,
    NoResetTotalFNMetric,
    NoResetTotalFPMetric,
    NoResetTotalTPMetric,
)
from pysatl_cpd.benchmark.online.noreset.metrics.online import (
    NoResetARLMetric,
    NoResetMeanDelayMetric,
    NoResetMedianDelayMetric,
)
from pysatl_cpd.benchmark.online.noreset.metrics.policy import (
    BisegmentPolicyBase,
    EventBasedPolicy,
    MixedPolicy,
    NoChangePolicy,
    NoResetPolicy,
    PointBasedPolicy,
)

NoResetTPMetric = NoResetTotalTPMetric

__all__ = [
    "BisegmentPolicyBase",
    "EventBasedPolicy",
    "MixedPolicy",
    "NoChangePolicy",
    "NoResetARLMetric",
    "NoResetClassificationReport",
    "NoResetDerivedMetric",
    "NoResetF1Metric",
    "NoResetMeanDelayMetric",
    "NoResetMedianDelayMetric",
    "NoResetMultipleRunMetric",
    "NoResetPolicy",
    "NoResetPrecisionMetric",
    "NoResetRecallMetric",
    "NoResetSingleRunMetric",
    "NoResetThresholdMetric",
    "NoResetTPMetric",
    "NoResetTotalFNMetric",
    "NoResetTotalFPMetric",
    "NoResetTotalTPMetric",
    "PointBasedPolicy",
    "wrap_noreset_derived_metric",
    "wrap_noreset_multiple_run_metric",
    "wrap_noreset_single_run_metric",
]
