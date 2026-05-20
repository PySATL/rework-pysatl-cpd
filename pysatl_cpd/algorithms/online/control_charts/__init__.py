# -*- coding: ascii -*-
"""Control chart algorithms for online change-point detection.

This subpackage provides online change-point detection algorithms based on
statistical process control chart techniques. Each algorithm processes
observations sequentially, maintaining running statistics and emitting a
scalar detection value that signals potential distributional shifts.

.. raw:: html

    <h2>Public API</h2>

- ``ShewhartControlChart``: Shewhart control chart with a sliding-window
  statistic. Tracks running mean and variance, computing a standardized
  deviation of the sliding-window mean from the global running mean.
- ``ShewhartControlChartConfiguration``: Frozen configuration dataclass for
  the Shewhart chart (learning period size, window size).
- ``ShewhartControlChartState``: Frozen state snapshot capturing running mean,
  variance, sample count, and sliding window contents at a given step.

.. raw:: html

    <h2>Subpackages</h2>

- ``visualizers``: Visualizers for rendering control chart algorithm state
  evolution over time. See that subpackage's docstring for details.

.. raw:: html

    <h2>Submodules</h2>

- ``shewhart_control_chart``: Contains the ``ShewhartControlChart`` class and
  its associated configuration and state dataclasses. See that module's
  docstring for the mathematical formulation and implementation details.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Run a Shewhart control chart on a synthetic data stream::

    >>> from pysatl_cpd.algorithms.online.control_charts import ShewhartControlChart
    >>> chart = ShewhartControlChart(learning_period_size=30, window_size=10)
    >>> values = [0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.1, -0.3, 0.0, 0.1,
    ...           0.2, -0.1, 0.0, 0.3, -0.2, 0.1, 0.0, -0.1, 0.2, 0.1,
    ...           -0.2, 0.0, 0.1, -0.1, 0.3, 0.0, -0.2, 0.1, 0.0, -0.1,
    ...           3.0, 3.2, 2.8, 3.1, 2.9, 3.0, 3.3, 2.7, 3.1, 2.8]
    >>> for v in values:
    ...     statistic = chart.process(v)
    >>> chart.state.samples_count
    40
    >>> chart.state.is_in_learning_period
    False

Use the chart with ``OnlineResetDetector`` for automatic reset-based detection::

    >>> from pysatl_cpd.algorithms.online.control_charts import ShewhartControlChart
    >>> from pysatl_cpd.core.online import OnlineResetDetector
    >>> from pysatl_cpd.data.generator import (
    ...     GenericSeriesGenerator,
    ...     NormalSpec,
    ...     ScenarioSpec,
    ...     SegmentPlan,
    ...     SegmentSpec,
    ...     build_plain_univariate_labeled_data,
    ... )
    >>> from pysatl_cpd.data.typedefs import StateDescriptor, frozendict
    >>> scenario = ScenarioSpec(
    ...     name="example",
    ...     segments=(
    ...         SegmentSpec(plan_name="base", length=100),
    ...         SegmentSpec(plan_name="shift", length=80),
    ...     ),
    ...     plans=frozendict(
    ...         base=SegmentPlan(
    ...             distribution=NormalSpec(mean=0.0, std=1.0),
    ...             state=StateDescriptor(type="base"),
    ...             name="base",
    ...         ),
    ...         shift=SegmentPlan(
    ...             distribution=NormalSpec(mean=3.0, std=1.0),
    ...             state=StateDescriptor(type="shift"),
    ...             name="shift",
    ...         ),
    ...     ),
    ... )
    >>> series = GenericSeriesGenerator(seed=42).generate_from_scenario(
    ...     scenario, name="example_series"
    ... )
    >>> provider = build_plain_univariate_labeled_data(
    ...     series, feature_name="value", name="example_provider"
    ... )
    >>> detector = OnlineResetDetector(
    ...     ShewhartControlChart(learning_period_size=30, window_size=10),
    ...     threshold=2.0,
    ...     skip_period=8,
    ...     collect_states=True,
    ... )
    >>> trace = detector.detect(provider)
    >>> len(trace.detected_change_points) > 0
    True

Inspect algorithm state after processing::

    >>> from pysatl_cpd.algorithms.online.control_charts import ShewhartControlChart
    >>> chart = ShewhartControlChart(learning_period_size=5, window_size=3)
    >>> for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
    ...     _ = chart.process(v)
    >>> state = chart.state
    >>> state.samples_count
    6
    >>> state.window_size
    3
    >>> round(state.mean, 2)
    3.5

Notes
-----
All algorithms in this subpackage implement the ``OnlineAlgorithm`` interface
from ``pysatl_cpd.core.online.ionline_algorithm``. They are designed to work
with ``OnlineResetDetector`` and ``OnlineCpdSolver`` from the core online
module for full detection pipelines.

The Shewhart control chart statistic is zero during the learning period and
when the running standard deviation is zero. The detection statistic follows
the formula ``S_t = |x_bar_w - mu| * sqrt(w) / sigma`` where ``x_bar_w`` is
the sliding-window mean, ``mu`` is the running mean, ``w`` is the window
size, and ``sigma`` is the running standard deviation.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.control_charts.shewhart_control_chart import (
    ShewhartControlChart,
    ShewhartControlChartConfiguration,
    ShewhartControlChartState,
)

__all__ = [
    "ShewhartControlChart",
    "ShewhartControlChartConfiguration",
    "ShewhartControlChartState",
]
