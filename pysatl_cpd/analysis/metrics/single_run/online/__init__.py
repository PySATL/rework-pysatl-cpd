# -*- coding: ascii -*-
"""Single-run online metrics.

This module provides metrics for evaluating online change point detection
algorithms on a single execution run. Each metric accepts a ``SingleRun``
pairing an ``OnlineDetectionTrace`` with a ``LabeledData`` provider and
returns a list of integer values, one per true change point or detection
event.

.. raw:: html

    <h2>Public API</h2>

- ``Delays`` -- computes the detection delay for each true change point.
  Missed detections are penalized with the maximum allowable delay.
  See :mod:`pysatl_cpd.analysis.metrics.single_run.online.delays` for details.
- ``RunLengths`` -- computes the distances between consecutive detected
  change points, starting from time step 0. See
  :mod:`pysatl_cpd.analysis.metrics.single_run.online.run_lengths` for details.

.. raw:: html

    <h2>Examples</h2>

Compute delays and run lengths for a single online detection run::

    >>> from pysatl_cpd.algorithms.online import ShewhartControlChart
    >>> from pysatl_cpd.analysis.metrics.single_run.online import Delays, RunLengths
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
    >>> delays = Delays(max_delay=20).evaluate(run)
    >>> run_lengths = RunLengths().evaluate(run)

.. raw:: html

    <h2>Notes</h2>

These metrics operate on a single run and return per-event lists. For
aggregated summaries across multiple runs (e.g., mean delay, median delay,
average run length), use the corresponding classes in
:mod:`pysatl_cpd.analysis.metrics.multiple_run.online`.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.single_run.online.delays import Delays
from pysatl_cpd.analysis.metrics.single_run.online.run_lengths import RunLengths

__all__ = [
    "Delays",
    "RunLengths",
]
