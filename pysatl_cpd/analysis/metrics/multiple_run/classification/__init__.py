# -*- coding: ascii -*-
"""Multiple-run classification metrics.

Micro-averaged classification metrics for evaluating change-point detection
algorithms across collections of runs. Each metric aggregates true-positive,
false-positive, and false-negative counts over all runs before computing
precision, recall, and F-beta scores.

The ``error_margin`` parameter on every metric defines an allowed (left, right)
tolerance window around each true change point for matching detections. A
detection falling within this window counts as a true positive; detections
outside any window are false positives, and true change points with no matching
detection are false negatives.

For the aggregation protocol and ``DerivedMetric`` base class, see the
``pysatl_cpd.analysis.metrics.multiple_run`` subpackage. For the underlying
single-run count metrics, see ``pysatl_cpd.analysis.metrics.single_run``.

.. raw:: html

    <h2>Public API</h2>

- ``TotalTP`` -- Sums true-positive counts across all runs. Inherits from
  ``TotalSum``. See :mod:`pysatl_cpd.analysis.metrics.multiple_run.classification.base`.
- ``TotalFP`` -- Sums false-positive counts across all runs. Inherits from
  ``TotalSum``. See :mod:`pysatl_cpd.analysis.metrics.multiple_run.classification.base`.
- ``TotalFN`` -- Sums false-negative counts across all runs. Inherits from
  ``TotalSum``. See :mod:`pysatl_cpd.analysis.metrics.multiple_run.classification.base`.
- ``PrecisionMetric`` -- Micro-averaged precision computed as total TP /
  (total TP + total FP). Returns 1.0 when the denominator is zero. See
  :mod:`pysatl_cpd.analysis.metrics.multiple_run.classification.precision`.
- ``RecallMetric`` -- Micro-averaged recall computed as total TP /
  (total TP + total FN). Returns 0.0 when the denominator is zero. See
  :mod:`pysatl_cpd.analysis.metrics.multiple_run.classification.recall`.
- ``FScoreMetric`` -- F-beta score derived from precision and recall.
  ``beta=1`` gives the F1 score. Raises ``ValueError`` if ``beta`` is
  negative. See :mod:`pysatl_cpd.analysis.metrics.multiple_run.classification.fmeasure`.
- ``ClassificationReport`` -- Returns a dictionary with keys ``tp``, ``fp``,
  ``fn``, ``precision``, ``recall``, and ``f1``. This is the most convenient
  entry point for a full classification summary. See
  :mod:`pysatl_cpd.analysis.metrics.multiple_run.classification.report`.

.. raw:: html

    <h2>Examples</h2>

All classification metrics require an ``error_margin`` tuple specifying the
allowed tolerance on each side of a true change point::

    >>> from pysatl_cpd.analysis.metrics.multiple_run.classification import (
    ...     ClassificationReport,
    ...     FScoreMetric,
    ...     PrecisionMetric,
    ...     RecallMetric,
    ...     TotalFN,
    ...     TotalFP,
    ...     TotalTP,
    ... )
    >>> report = ClassificationReport(error_margin=(0, 15))  # doctest: +SKIP
    >>> precision = PrecisionMetric(error_margin=(0, 15))  # doctest: +SKIP
    >>> recall = RecallMetric(error_margin=(0, 15))  # doctest: +SKIP
    >>> f1 = FScoreMetric(error_margin=(0, 15))  # doctest: +SKIP
    >>> tp = TotalTP(error_margin=(0, 15))  # doctest: +SKIP
    >>> fp = TotalFP(error_margin=(0, 15))  # doctest: +SKIP
    >>> fn = TotalFN(error_margin=(0, 15))  # doctest: +SKIP

Evaluate on a sequence of ``SingleRun`` objects to get aggregated results::

    >>> from pysatl_cpd.core.single_run import SingleRun
    >>> # runs: Sequence[SingleRun[DetectionTrace, LabeledData]]
    >>> result = ClassificationReport(error_margin=(0, 15)).evaluate(runs)  # doctest: +SKIP
    >>> result["precision"]  # doctest: +SKIP
    0.895
    >>> result["recall"]  # doctest: +SKIP
    0.935
    >>> result["f1"]  # doctest: +SKIP
    0.914

Use a custom beta value to weight recall more heavily than precision::

    >>> f2 = FScoreMetric(error_margin=(0, 15), beta=2.0)  # doctest: +SKIP
    >>> f2.evaluate(runs)  # doctest: +SKIP
    0.925...

Use metrics through the reset benchmark API::

    >>> from pysatl_cpd.analysis.metrics.multiple_run.classification import ClassificationReport
    >>> from pysatl_cpd.benchmark.online.reset import OnlineResetBenchmark
    >>> benchmark = OnlineResetBenchmark(dataset=dataset, registry=registry)  # doctest: +SKIP
    >>> results = benchmark.get_metrics_for(entries, ClassificationReport(error_margin=(0, 15)))  # doctest: +SKIP

.. raw:: html

    <h2>Notes</h2>

- All metrics in this module perform micro-averaging: counts are summed across
  runs before the ratio is computed. This is different from macro-averaging,
  which would compute per-run ratios and then average them.
- ``PrecisionMetric`` returns 1.0 when TP + FP is zero (no detections made).
  ``RecallMetric`` returns 0.0 when TP + FN is zero (no true change points).
  ``FScoreMetric`` returns 0.0 when the weighted denominator is zero.
- ``ClassificationReport`` is typically the most convenient entry point. Use
  the individual metrics (``PrecisionMetric``, ``RecallMetric``, ``FScoreMetric``,
  ``TotalTP``, ``TotalFP``, ``TotalFN``) when you need specific components
  without computing the full report.
- For online-specific metrics such as detection delay and average run length,
  see :mod:`pysatl_cpd.analysis.metrics.multiple_run.online`.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.multiple_run.classification.base import TotalFN, TotalFP, TotalTP
from pysatl_cpd.analysis.metrics.multiple_run.classification.fmeasure import FScoreMetric
from pysatl_cpd.analysis.metrics.multiple_run.classification.precision import PrecisionMetric
from pysatl_cpd.analysis.metrics.multiple_run.classification.recall import RecallMetric
from pysatl_cpd.analysis.metrics.multiple_run.classification.report import ClassificationReport

__all__ = [
    "ClassificationReport",
    "FScoreMetric",
    "PrecisionMetric",
    "RecallMetric",
    "TotalFN",
    "TotalFP",
    "TotalTP",
]
