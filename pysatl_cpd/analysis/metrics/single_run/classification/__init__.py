# -*- coding: ascii -*-
"""Single-run classification metrics.

Count-based metrics that evaluate one detection run by matching detected
change points to ground-truth change points within a configurable error
margin. Each metric returns an integer count (true positives, false
positives, or false negatives) for a single ``SingleRun``.

The matching algorithm assigns each true change point the earliest
unmatched detection that falls within its ``(left, right)`` tolerance
window. Detections that match no true change point are false positives;
true change points with no matching detection are false negatives.

These primitives are the building blocks for higher-level metrics. For
aggregated precision, recall, and F-scores across multiple runs, see
``pysatl_cpd.analysis.metrics.multiple_run.classification``.

.. raw:: html

    <h2>Public API</h2>

- ``ClassificationMetricBase`` -- Generic base class for single-run
  metrics built on change-point matching. Provides the ``match()``
  method and ``error_margin`` validation. See
  :mod:`pysatl_cpd.analysis.metrics.single_run.classification.base`.
- ``ClassificationPrimitive`` -- Generic base class for count-based
  single-run classification metrics. Subclasses return ``int`` from
  ``evaluate()``. See
  :mod:`pysatl_cpd.analysis.metrics.single_run.classification.base`.
- ``TruePositiveCount`` -- Counts true change points that have at least
  one matched detection. See
  :mod:`pysatl_cpd.analysis.metrics.single_run.classification.tp_metric`.
- ``FalsePositiveCount`` -- Counts detections that are not matched to
  any true change point. See
  :mod:`pysatl_cpd.analysis.metrics.single_run.classification.fp_metric`.
- ``FalseNegativeCount`` -- Counts true change points that have no
  matched detection. See
  :mod:`pysatl_cpd.analysis.metrics.single_run.classification.fn_metric`.

.. raw:: html

    <h2>Examples</h2>

Instantiate a count metric with an error margin tuple ``(left, right)``::

    >>> from pysatl_cpd.analysis.metrics.single_run.classification import (
    ...     FalseNegativeCount,
    ...     FalsePositiveCount,
    ...     TruePositiveCount,
    ... )
    >>> tp = TruePositiveCount(error_margin=(0, 15))
    >>> fp = FalsePositiveCount(error_margin=(0, 15))
    >>> fn = FalseNegativeCount(error_margin=(0, 15))

Evaluate on a ``SingleRun`` to get the integer count::

    >>> from pysatl_cpd.core.single_run import SingleRun
    >>> # run: SingleRun[DetectionTrace, LabeledData]
    >>> tp_count = tp.evaluate(run)  # doctest: +SKIP
    >>> fp_count = fp.evaluate(run)  # doctest: +SKIP
    >>> fn_count = fn.evaluate(run)  # doctest: +SKIP

Use the base-class ``match()`` method directly to inspect the matching::

    >>> from pysatl_cpd.analysis.metrics.single_run.classification.base import (
    ...     ClassificationMetricBase,
    ... )
    >>> metric = ClassificationMetricBase(error_margin=(2, 2))  # doctest: +SKIP
    >>> matching = metric.match([8, 12, 25], [10, 24])  # doctest: +SKIP
    >>> matching  # doctest: +SKIP
    {10: {8, 12}, 24: {25}}

.. raw:: html

    <h2>Notes</h2>

- ``error_margin`` must be a tuple of two non-negative integers. A
  ``ValueError`` is raised if either component is negative.
- Matching is greedy and left-to-right: true change points are processed
  in sorted order, and each claims the earliest unmatched detection within
  its tolerance window.
- All classes in this module are generic over ``TraceT`` (a
  ``DetectionTrace`` subtype) and ``ProviderT`` (a ``LabeledData``
  subtype). The type parameters are inferred from the ``SingleRun``
  passed to ``evaluate()``.
- For multi-run aggregation (precision, recall, F-score), use the
  metrics in ``pysatl_cpd.analysis.metrics.multiple_run.classification``.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.analysis.metrics.single_run.classification.base import ClassificationMetricBase, ClassificationPrimitive
from pysatl_cpd.analysis.metrics.single_run.classification.fn_metric import FalseNegativeCount
from pysatl_cpd.analysis.metrics.single_run.classification.fp_metric import FalsePositiveCount
from pysatl_cpd.analysis.metrics.single_run.classification.tp_metric import TruePositiveCount

__all__ = [
    "ClassificationMetricBase",
    "ClassificationPrimitive",
    "FalseNegativeCount",
    "FalsePositiveCount",
    "TruePositiveCount",
]
