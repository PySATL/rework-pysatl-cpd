# -*- coding: ascii -*-
"""Single-run metrics for evaluating one detection trace against labeled data.

This module provides the base interface and concrete metrics that evaluate a
single execution of a change-point detector. Each metric operates on one
``SingleRun``, which pairs a ``DetectionTrace`` (or ``OnlineDetectionTrace``)
with a ``LabeledData`` provider containing ground-truth change points.

The module is organized into two subpackages:

- ``classification`` -- count-based metrics that match detected change points
  to ground truth within a configurable error margin. See the subpackage
  docstring for details and examples.
- ``online`` -- timing-based metrics for online detectors, including detection
  delays and run lengths between consecutive alarms. See the subpackage
  docstring for details and examples.

.. raw:: html

    <h2>Public API</h2>

- ``ISingleRunMetric`` -- abstract base interface for all single-run metrics.
  Generic over ``TraceT``, ``ProviderT``, and ``ResultT``. See
  :mod:`pysatl_cpd.analysis.metrics.abstracts.isingle_run_metric`.
- ``ClassificationPrimitive`` -- base class for count-based classification
  metrics (TP, FP, FN). See
  :mod:`pysatl_cpd.analysis.metrics.single_run.classification`.
- ``TruePositiveCount`` -- counts true change points with at least one matched
  detection. See
  :mod:`pysatl_cpd.analysis.metrics.single_run.classification`.
- ``FalsePositiveCount`` -- counts detections not matched to any true change
  point. See
  :mod:`pysatl_cpd.analysis.metrics.single_run.classification`.
- ``FalseNegativeCount`` -- counts true change points with no matched
  detection. See
  :mod:`pysatl_cpd.analysis.metrics.single_run.classification`.
- ``Delays`` -- computes per-change-point detection delays for online
  algorithms. See :mod:`pysatl_cpd.analysis.metrics.single_run.online`.
- ``RunLengths`` -- computes distances between consecutive detections for
  online algorithms. See :mod:`pysatl_cpd.analysis.metrics.single_run.online`.

Examples
--------
Evaluate classification primitives on a single run::

    >>> from pysatl_cpd.algorithms.online import ShewhartControlChart
    >>> from pysatl_cpd.analysis.metrics.single_run import (
    ...     Delays,
    ...     FalseNegativeCount,
    ...     FalsePositiveCount,
    ...     RunLengths,
    ...     TruePositiveCount,
    ... )
    >>> from pysatl_cpd.core.online import OnlineResetDetector
    >>> from pysatl_cpd.core.single_run import SingleRun
    >>> from pysatl_cpd.data.generator import (
    ...     GenericSeriesGenerator,
    ...     ScenarioSpec,
    ...     SegmentPlan,
    ...     SegmentSpec,
    ...     UnivariateDistributionSpec,
    ...     build_plain_univariate_labeled_data,
    ... )
    >>> from pysatl_cpd.data.typedefs import frozendict
    >>> scenario = ScenarioSpec(
    ...     name="example",
    ...     segments=(
    ...         SegmentSpec(plan_name="baseline", length=100),
    ...         SegmentSpec(plan_name="shifted", length=100),
    ...     ),
    ...     plans=frozendict(
    ...         baseline=SegmentPlan(
    ...             distribution=UnivariateDistributionSpec("Normal", "meanStd", mu=0.0, sigma=1.0)
    ...         ),
    ...         shifted=SegmentPlan(
    ...             distribution=UnivariateDistributionSpec("Normal", "meanStd", mu=2.0, sigma=1.0)
    ...         ),
    ...     ),
    ... )
    >>> series = GenericSeriesGenerator(seed=42).generate_from_scenario(scenario)
    >>> provider = build_plain_univariate_labeled_data(series, feature_name="value", name="example")
    >>> detector = OnlineResetDetector(ShewhartControlChart(learning_period_size=30, window_size=10), threshold=3.0)
    >>> trace = detector.detect(provider)
    >>> run = SingleRun(trace=trace, provider=provider)
    >>> error_margin = (0, 15)
    >>> tp = TruePositiveCount(error_margin=error_margin).evaluate(run)
    >>> fp = FalsePositiveCount(error_margin=error_margin).evaluate(run)
    >>> fn = FalseNegativeCount(error_margin=error_margin).evaluate(run)
    >>> delays = Delays(max_delay=error_margin[1]).evaluate(run)
    >>> run_lengths = RunLengths().evaluate(run)

Notes
-----
- All concrete metrics implement ``ISingleRunMetric.evaluate()``, which
  accepts a ``SingleRun`` and returns a result (``int`` for classification
  primitives, ``list[int]`` for online metrics).
- Classification metrics require an ``error_margin`` tuple of two
  non-negative integers ``(left, right)``. A ``ValueError`` is raised if
  either component is negative.
- ``Delays`` requires a ``max_delay`` that must be non-negative. Missed
  detections are penalized with this maximum value.
- ``RunLengths`` does not use ground truth; it measures intervals between
  consecutive detections starting from time step 0.
- For aggregated metrics across multiple runs (precision, recall, F1, mean
  delay, median delay, ARL), see
  :mod:`pysatl_cpd.analysis.metrics.multiple_run`.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.analysis.metrics.abstracts.isingle_run_metric import ISingleRunMetric
from pysatl_cpd.analysis.metrics.single_run.classification.base import ClassificationPrimitive
from pysatl_cpd.analysis.metrics.single_run.classification.fn_metric import FalseNegativeCount
from pysatl_cpd.analysis.metrics.single_run.classification.fp_metric import FalsePositiveCount
from pysatl_cpd.analysis.metrics.single_run.classification.tp_metric import TruePositiveCount
from pysatl_cpd.analysis.metrics.single_run.online.delays import Delays
from pysatl_cpd.analysis.metrics.single_run.online.run_lengths import RunLengths

__all__ = [
    "Delays",
    "RunLengths",
    "ClassificationPrimitive",
    "FalseNegativeCount",
    "FalsePositiveCount",
    "TruePositiveCount",
    "ISingleRunMetric",
]
