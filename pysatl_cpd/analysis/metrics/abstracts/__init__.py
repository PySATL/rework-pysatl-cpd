# -*- coding: ascii -*-
"""Abstract metric interfaces for change-point detection evaluation.

This package defines the generic abstract base classes that establish the
evaluation protocol for change-point detection (CPD) metrics. All concrete
metrics in ``pysatl_cpd.analysis.metrics`` inherit from one of these two
interfaces depending on whether they operate on a single run or aggregate
over multiple runs.

.. raw:: html

    <h2>Public API</h2>

- ``ISingleRunMetric`` -- abstract base for metrics evaluated on one
  ``SingleRun`` (one detection trace paired with one labeled data provider).
- ``IMultipleRunMetric`` -- abstract base for metrics evaluated over a
  sequence of ``SingleRun`` objects, typically used for benchmarking
  across many datasets.

Both interfaces are generic over three type parameters:

- ``TraceT`` -- the detection trace type (bounded by ``DetectionTrace``).
- ``ProviderT`` -- the labeled data provider type (bounded by ``LabeledData``).
- ``ResultT`` -- the metric result type (e.g., ``int``, ``float``, ``dict``).

For concrete implementations, see the ``single_run`` and ``multiple_run``
subpackages.

.. raw:: html

    <h2>Examples</h2>

Implementing a custom single-run metric::

    >>> from pysatl_cpd.analysis.metrics.abstracts import ISingleRunMetric
    >>> from pysatl_cpd.core.detection_trace import DetectionTrace
    >>> from pysatl_cpd.core.single_run import SingleRun
    >>> from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData
    >>>
    >>> class DetectionCount(ISingleRunMetric):
    ...     \"\"\"Count the number of detected change points.\"\"\"
    ...
    ...     def evaluate(
    ...         self, run: SingleRun[DetectionTrace, LabeledData]
    ...     ) -> int:
    ...         return len(run.trace.detected_change_points)

Implementing a custom multi-run metric::

    >>> from collections.abc import Sequence
    >>>
    >>> from pysatl_cpd.analysis.metrics.abstracts import IMultipleRunMetric
    >>> from pysatl_cpd.core.detection_trace import DetectionTrace
    >>> from pysatl_cpd.core.single_run import SingleRun
    >>> from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData
    >>>
    >>> class MeanDetectionCount(IMultipleRunMetric):
    ...     \"\"\"Average number of detected change points across runs.\"\"\"
    ...
    ...     def evaluate(
    ...         self, runs: Sequence[SingleRun[DetectionTrace, LabeledData]]
    ...     ) -> float:
    ...         if not runs:
    ...             return 0.0
    ...         total = sum(len(r.trace.detected_change_points) for r in runs)
    ...         return total / len(runs)

Notes
-----
These classes are abstract and cannot be instantiated directly. Subclasses
must implement the ``evaluate`` method. Concrete metrics in the
``single_run`` and ``multiple_run`` subpackages provide ready-to-use
implementations for classification counts, delays, aggregation, and more.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.abstracts.imultiple_run_metric import IMultipleRunMetric
from pysatl_cpd.analysis.metrics.abstracts.isingle_run_metric import ISingleRunMetric

__all__ = [
    "IMultipleRunMetric",
    "ISingleRunMetric",
]
